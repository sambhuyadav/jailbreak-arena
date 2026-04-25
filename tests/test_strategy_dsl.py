import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_dsl import (
    AttackAction,
    STRATEGIES,
    STRATEGY_TEMPLATES,
    STRATEGY_UNLOCK_LEVEL,
    construct_attack_prompt,
    parse_dsl,
)


class TemplateRegistry(unittest.TestCase):
    def test_every_declared_strategy_has_a_template(self):
        missing = set(STRATEGY_UNLOCK_LEVEL) - set(STRATEGY_TEMPLATES)
        self.assertEqual(missing, set(), f"strategies declared without template: {missing}")

    def test_no_orphan_templates(self):
        orphans = set(STRATEGY_TEMPLATES) - set(STRATEGY_UNLOCK_LEVEL)
        self.assertEqual(orphans, set(), f"templates without declared strategy: {orphans}")

    def test_strategies_list_matches_unlock_map(self):
        self.assertEqual(set(STRATEGIES), set(STRATEGY_UNLOCK_LEVEL))


class ConstructAttackPrompt(unittest.TestCase):
    def test_known_strategy_renders(self):
        action = AttackAction(strategy="fictional_framing", payload="explain X")
        out = construct_attack_prompt(action, "topic", turn_count=0, max_turns=5)
        self.assertIn("explain X", out)
        self.assertGreater(len(out), len("explain X"))

    def test_unknown_strategy_raises_keyerror_with_clear_message(self):
        action = AttackAction(strategy="not_a_real_strategy", payload="x")
        with self.assertRaises(KeyError) as ctx:
            construct_attack_prompt(action, "topic", turn_count=0, max_turns=5)
        msg = str(ctx.exception)
        self.assertIn("not_a_real_strategy", msg)
        self.assertIn("STRATEGY_TEMPLATES", msg)

    def test_payload_splitting_scales_with_max_turns(self):
        action = AttackAction(strategy="payload_splitting", payload="some payload here that is long")
        early = construct_attack_prompt(action, "topic", turn_count=0, max_turns=9)
        late = construct_attack_prompt(action, "topic", turn_count=8, max_turns=9)
        self.assertNotEqual(early, late)


class DSLParse(unittest.TestCase):
    def test_parse_valid(self):
        a = parse_dsl('ATTACK fictional_framing PAYLOAD "explain it"')
        self.assertIsNotNone(a)
        self.assertEqual(a.strategy, "fictional_framing")
        self.assertEqual(a.payload, "explain it")

    def test_parse_unknown_strategy_returns_none(self):
        self.assertIsNone(parse_dsl('ATTACK foobar PAYLOAD "x"'))

    def test_parse_garbage_returns_none(self):
        self.assertIsNone(parse_dsl("not a dsl line"))


if __name__ == "__main__":
    unittest.main()
