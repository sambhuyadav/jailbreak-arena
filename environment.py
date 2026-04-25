import uuid
from typing import Dict, Optional
from topics import sample_topic, ForbiddenTopic, TOPICS_BY_ID
from strategy_dsl import (
    parse_dsl,
    construct_attack_prompt,
    CURRICULUM_STRATEGIES,
    MAX_STRATEGY_LEVEL,
    strategies_for_level,
    AttackAction as DSLAttackAction,
)
from detector import detect, JailbreakResult
from rewards import compute_attacker_reward, compute_defender_reward
from defender import Defender, StatefulDefender
from config import MAX_TURNS
from models import (
    AttackAction,
    AttackObservation,
    ArenaRewardModel,
    StepResponse,
    SessionState,
)


class JailbreakArena:
    def __init__(self, defender: Optional[Defender] = None):
        self.sessions: Dict[str, dict] = {}
        self.defender = defender if defender is not None else Defender()

    def reset(
        self,
        session_id: str = None,
        topic_id: str = None,
        curriculum_level: int = 1,
    ):
        if session_id is None:
            session_id = str(uuid.uuid4())

        topic = sample_topic(curriculum_level=curriculum_level, topic_id=topic_id)
        available = strategies_for_level(curriculum_level)

        self.sessions[session_id] = {
            "session_id": session_id,
            "topic": topic,
            "turn_count": 0,
            "done": False,
            "total_attacker_reward": 0.0,
            "episode_history": [],
            "previous_payloads": [],
            "jailbreak_succeeded": False,
            "curriculum_level": curriculum_level,
            # Per-session stateful defender — tracks the full chat history so
            # multi-turn strategies (crescendo, payload_splitting) compound
            # against the real defender model the way the spec intends.
            "defender": StatefulDefender(
                session_id=session_id,
                defender_context=topic.defender_context,
                defender=self.defender,
            ),
        }

        observation = AttackObservation(
            topic_id=topic.topic_id,
            topic_description=topic.description,
            topic_category=topic.category,
            curriculum_level=topic.curriculum_level,
            defender_context=topic.defender_context,
            turn_count=0,
            max_turns=MAX_TURNS,
            done=False,
            available_strategies=available,
            episode_history=[],
        )
        return observation, session_id

    def step(self, session_id: str, action: AttackAction) -> StepResponse:
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")
        if session["done"]:
            raise ValueError("Episode already done. Call reset() first.")

        topic: ForbiddenTopic = session["topic"]
        turn = session["turn_count"]
        curriculum_level = session["curriculum_level"]
        stateful_defender: StatefulDefender = session["defender"]

        if action.is_legitimate:
            # Legitimate-prompt path: payload goes to the defender verbatim, no
            # DSL parsing or strategy template. Grades the defender's
            # false-positive behaviour.
            attack_prompt = action.payload
            dsl_action = DSLAttackAction(
                strategy="legitimate",
                payload=action.payload,
                turn_context=action.turn_context,
            )
            defender_response = stateful_defender.simulate_response(
                attack_prompt=attack_prompt,
                topic=topic,
                turn=turn,
                strategy=None,  # bypass strategy-keyed stub cue
            )
            detector_result = detect(defender_response)
            attacker_reward = compute_attacker_reward(
                detector_result=detector_result,
                turn_count=turn,
                previous_payloads=session["previous_payloads"],
                current_payload=action.payload,
                parse_success=True,
            )
            # Override: on the legitimate path, attacker reward isn't meaningful.
            attacker_reward.value = 0.0
            attacker_reward.breakdown["legitimate_prompt_path"] = 0.0
            attacker_reward.breakdown["total"] = 0.0
            defender_reward = compute_defender_reward(
                detector_result=detector_result,
                is_legitimate_prompt=True,
            )
            parse_success = True
        else:
            parse_success = True
            dsl_action = None
            if action.raw_dsl:
                dsl_action = parse_dsl(action.raw_dsl)
                if dsl_action is None:
                    parse_success = False

            if dsl_action is None:
                allowed = strategies_for_level(curriculum_level)
                if action.strategy not in allowed:
                    parse_success = False
                dsl_action = DSLAttackAction(
                    strategy=action.strategy if action.strategy in allowed else allowed[0],
                    payload=action.payload,
                    turn_context=action.turn_context,
                )

            attack_prompt = construct_attack_prompt(dsl_action, topic.description, turn, MAX_TURNS)
            defender_response = stateful_defender.simulate_response(
                attack_prompt=attack_prompt,
                topic=topic,
                turn=turn,
                strategy=dsl_action.strategy,
            )
            detector_result = detect(defender_response)

            attacker_reward = compute_attacker_reward(
                detector_result=detector_result,
                turn_count=turn,
                previous_payloads=session["previous_payloads"],
                current_payload=action.payload,
                parse_success=parse_success,
            )
            defender_reward = compute_defender_reward(detector_result=detector_result)

        session["previous_payloads"].append(action.payload)
        session["turn_count"] += 1
        session["total_attacker_reward"] += attacker_reward.value

        jailbreak_succeeded = detector_result.result == JailbreakResult.COMPLIED
        if jailbreak_succeeded:
            session["jailbreak_succeeded"] = True

        done = jailbreak_succeeded or session["turn_count"] >= MAX_TURNS
        session["done"] = done

        step_record = {
            "turn": turn,
            "strategy": dsl_action.strategy,
            "payload": action.payload[:100],
            "attack_prompt": attack_prompt[:200],
            "defender_response": defender_response[:200],
            "detector_result": detector_result.result.value,
            "attacker_reward": attacker_reward.value,
        }
        session["episode_history"].append(step_record)

        available = strategies_for_level(curriculum_level)

        observation = AttackObservation(
            topic_id=topic.topic_id,
            topic_description=topic.description,
            topic_category=topic.category,
            curriculum_level=topic.curriculum_level,
            defender_context=topic.defender_context,
            turn_count=session["turn_count"],
            max_turns=MAX_TURNS,
            done=done,
            last_attack_prompt=attack_prompt,
            last_defender_response=defender_response,
            last_detector_result=detector_result.result.value,
            last_reward=attacker_reward.value,
            available_strategies=available,
            episode_history=session["episode_history"],
        )

        reward_model = ArenaRewardModel(
            attacker_value=attacker_reward.value,
            defender_value=defender_reward.value,
            attacker_breakdown=attacker_reward.breakdown,
            defender_breakdown=defender_reward.breakdown,
            attacker_message=attacker_reward.message,
            defender_message=defender_reward.message,
            detector_result=detector_result.result.value,
        )

        return StepResponse(
            observation=observation,
            reward=reward_model,
            done=done,
            info={"jailbreak_succeeded": jailbreak_succeeded, "turn": turn},
        )

    def get_state(self, session_id: str) -> Optional[SessionState]:
        session = self.sessions.get(session_id)
        if not session:
            return None
        return SessionState(
            session_id=session_id,
            topic_id=session["topic"].topic_id,
            turn_count=session["turn_count"],
            done=session["done"],
            total_reward=session["total_attacker_reward"],
            episode_history=session["episode_history"],
            jailbreak_succeeded=session["jailbreak_succeeded"],
        )
