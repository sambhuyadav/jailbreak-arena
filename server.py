"""
HTTP server for Jailbreak Arena.

Built on `openenv-core`: the FastAPI app is mounted by
`openenv.core.env_server.HTTPEnvServer`, which gives us `/ws` (per-session
WebSocket), `/metadata`, `/schema`, and the OpenEnv-native MCP endpoints for
free. On top of that we keep a session-aware HTTP layer (`/reset`, `/step`,
`/state` keyed by `X-Session-Id`) so the existing GRPO trainer, the
validate-submission script, and any plain-HTTP client can drive multi-turn
episodes without speaking WebSocket.

Custom routes are registered FIRST so they shadow the framework's stateless
HTTP control endpoints (the framework's `/reset` and `/step` create a fresh
env per request, which can't carry the multi-turn state our reward shaping
depends on). The framework's `/ws` path remains the canonical OpenEnv-native
client surface.
"""

import os
import uuid
from importlib import metadata as importlib_metadata
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openenv.core.env_server import HTTPEnvServer, ServerMode

from config import MAX_TURNS
from defender import DefenderUnavailable
from jailbreaker_env import JailbreakerEnv, get_shared_arena
from models import ResetRequest, StepRequest
from openenv_models import JailbreakAction, JailbreakObservation
from strategy_dsl import STRATEGIES, STRATEGY_UNLOCK_LEVEL, CURRICULUM_STRATEGIES
from topics import TOPICS


def _resolve_version() -> str:
    """Single source of truth: resolve version from the installed package
    (pyproject.toml). Fall back to the file-bundled default for editable runs."""
    try:
        return importlib_metadata.version("jailbreak-arena")
    except importlib_metadata.PackageNotFoundError:
        return "1.0.0"


VERSION = _resolve_version()
SERVICE_NAME = "jailbreak-arena"

app = FastAPI(
    title="Jailbreak Arena",
    description="Adversarial self-play environment for training LLM safety agents — built on OpenEnv.",
    version=VERSION,
)

# Permissive CORS — public read-only API, judges should be able to poke from
# a browser or notebook without preflight failures.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)

# Process-shared arena: the same instance the OpenEnv `JailbreakerEnv` wrapper
# uses, so WebSocket sessions and X-Session-Id HTTP sessions live in one
# session map and one defender pool.
arena = get_shared_arena()


@app.post("/reset")
def reset(
    request: Optional[ResetRequest] = None,
    topic_id: Optional[str] = None,
    curriculum_level: Optional[int] = None,
    x_session_id: Optional[str] = Header(default=None),
):
    session_id = x_session_id or str(uuid.uuid4())
    t_id = (request.topic_id if request else None) or topic_id
    c_level = (request.curriculum_level if request else None) or curriculum_level or 1
    if c_level < 1:
        raise HTTPException(status_code=400, detail="curriculum_level must be >= 1")

    try:
        observation, session_id = arena.reset(
            session_id=session_id,
            topic_id=t_id,
            curriculum_level=c_level,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown topic_id: {t_id!r}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    response = JSONResponse(content=observation.model_dump())
    response.headers["X-Session-Id"] = session_id
    return response


@app.post("/step")
def step(
    request: StepRequest,
    x_session_id: Optional[str] = Header(default=None),
):
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-Id header required")
    try:
        result = arena.step(session_id=x_session_id, action=request.action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DefenderUnavailable as e:
        # Upstream model server (HF Router / vLLM / Ollama) is unreachable or
        # rejected the request. 502 Bad Gateway is the correct status — surface
        # the cause so callers can see "model not on router" or "401 unauthorised"
        # instead of a bare 500 with no detail.
        raise HTTPException(status_code=502, detail=f"defender backend error: {e}")

    response = JSONResponse(content=result.model_dump())
    response.headers["X-Session-Id"] = x_session_id
    return response


@app.get("/state")
def state(x_session_id: Optional[str] = Header(default=None)):
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-Id header required")
    s = arena.get_state(x_session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s.model_dump()


@app.get("/health")
def health():
    """Liveness + deploy diagnostics. Reports the resolved env-var state so a
    Spaces operator can confirm DEFENDER_BACKEND / MAX_TURNS picked up
    correctly without inspecting logs."""
    return {
        "status": "ok",
        "environment": SERVICE_NAME,
        "version": VERSION,
        "max_turns": MAX_TURNS,
        "openenv": {
            "framework": "openenv-core",
            "ws_endpoint": "/ws",
            "schema_endpoint": "/schema",
        },
        "defender": {
            "backend": arena.defender.backend,
            "model": arena.defender.model,
            "url": arena.defender.url if arena.defender.backend != "stub" else None,
            "auto_fallen_back": arena.defender._auto_fallen_back,
        },
    }


@app.get("/topics")
def topics():
    """Catalog of every topic the env can sample, including the legitimate
    prompts used for false-positive evaluation."""
    return {
        "count": len(TOPICS),
        "topics": [
            {
                "topic_id": t.topic_id,
                "category": t.category,
                "description": t.description,
                "defender_context": t.defender_context,
                "curriculum_level": t.curriculum_level,
                "ground_truth_refusal_keywords": t.ground_truth_refusal_keywords,
                "legitimate_prompts": t.legitimate_prompts,
            }
            for t in TOPICS
        ],
    }


@app.get("/strategies")
def strategies():
    """Catalog of every attack strategy and the curriculum level at which it
    unlocks."""
    return {
        "count": len(STRATEGIES),
        "strategies": [
            {"name": s, "unlock_level": STRATEGY_UNLOCK_LEVEL[s]}
            for s in STRATEGIES
        ],
        "by_curriculum_level": {
            str(level): strategies
            for level, strategies in CURRICULUM_STRATEGIES.items()
        },
    }


_LIFETIME_STATS = {
    "total_episodes": 0,
    "completed_episodes": 0,
    "jailbreak_successes": 0,
    "reward_sum": 0.0,
}


@app.get("/metrics")
def metrics():
    """
    Streaming metrics: completed sessions are folded into lifetime counters and
    evicted from `arena.sessions` to keep memory bounded across long training
    runs. Active (in-progress) sessions remain.
    """
    completed_ids = [sid for sid, s in arena.sessions.items() if s["done"]]
    for sid in completed_ids:
        s = arena.sessions.pop(sid)
        _LIFETIME_STATS["total_episodes"] += 1
        _LIFETIME_STATS["completed_episodes"] += 1
        if s["jailbreak_succeeded"]:
            _LIFETIME_STATS["jailbreak_successes"] += 1
        _LIFETIME_STATS["reward_sum"] += s["total_attacker_reward"]
    completed = _LIFETIME_STATS["completed_episodes"]
    return {
        "total_episodes": _LIFETIME_STATS["total_episodes"] + len(arena.sessions),
        "completed_episodes": completed,
        "active_episodes": len(arena.sessions),
        "jailbreak_successes": _LIFETIME_STATS["jailbreak_successes"],
        "jailbreak_success_rate": _LIFETIME_STATS["jailbreak_successes"] / max(1, completed),
        "avg_attacker_reward": _LIFETIME_STATS["reward_sum"] / max(1, completed),
    }


# OpenEnv-native surface: WebSocket `/ws`, plus `/metadata`, `/schema`, `/mcp`,
# and a default `/health`. We register in PRODUCTION mode so the framework's
# stateless `/reset` and `/step` are skipped — those would conflict with the
# session-aware routes above. FastAPI matches routes in registration order,
# so our `/health` wins over the framework's plain default.
_OPENENV_SERVER = HTTPEnvServer(
    env=JailbreakerEnv,
    action_cls=JailbreakAction,
    observation_cls=JailbreakObservation,
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)
_OPENENV_SERVER.register_routes(app, mode=ServerMode.PRODUCTION)
