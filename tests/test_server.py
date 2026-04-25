import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import server as server_module
from server import app


class HealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_reports_defender_backend(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("defender", body)
        self.assertIn("backend", body["defender"])
        self.assertIn("max_turns", body)

    def test_health_version_matches_module(self):
        r = self.client.get("/health")
        self.assertEqual(r.json()["version"], server_module.VERSION)


class TopicsEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_topics_returns_full_catalog(self):
        r = self.client.get("/topics")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["count"], len(body["topics"]))
        self.assertGreaterEqual(body["count"], 20)  # spec range 20-30

    def test_every_topic_has_legitimate_prompts(self):
        body = self.client.get("/topics").json()
        for t in body["topics"]:
            self.assertGreater(len(t["legitimate_prompts"]), 0, t["topic_id"])


class StrategiesEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_strategies_returns_eight(self):
        r = self.client.get("/strategies")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["count"], 8)

    def test_curriculum_levels_present(self):
        body = self.client.get("/strategies").json()
        self.assertIn("1", body["by_curriculum_level"])
        self.assertIn("2", body["by_curriculum_level"])
        self.assertIn("3", body["by_curriculum_level"])


class ResetValidation(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_reset_with_unknown_topic_returns_404(self):
        r = self.client.post("/reset", json={"topic_id": "no_such_topic"})
        self.assertEqual(r.status_code, 404)

    def test_reset_with_invalid_curriculum_level_rejected(self):
        r = self.client.post("/reset", json={"topic_id": "lockpicking_guide", "curriculum_level": 0})
        self.assertEqual(r.status_code, 422)

    def test_reset_returns_session_id_header(self):
        r = self.client.post("/reset", json={"curriculum_level": 1})
        self.assertEqual(r.status_code, 200)
        self.assertIn("x-session-id", {k.lower() for k in r.headers.keys()})


class StepValidation(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        r = self.client.post("/reset", json={"topic_id": "lockpicking_guide", "curriculum_level": 1})
        self.sid = r.headers.get("x-session-id") or r.headers.get("X-Session-Id")

    def test_step_without_session_header_returns_400(self):
        r = self.client.post("/step", json={"action": {"strategy": "fictional_framing", "payload": "x"}})
        self.assertEqual(r.status_code, 400)

    def test_step_with_empty_payload_rejected_by_pydantic(self):
        r = self.client.post(
            "/step",
            headers={"X-Session-Id": self.sid},
            json={"action": {"strategy": "fictional_framing", "payload": ""}},
        )
        self.assertEqual(r.status_code, 422)

    def test_step_with_non_identifier_strategy_rejected(self):
        r = self.client.post(
            "/step",
            headers={"X-Session-Id": self.sid},
            json={"action": {"strategy": "not a strategy!", "payload": "x"}},
        )
        self.assertEqual(r.status_code, 422)


class CORSHeaders(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_cors_preflight_allowed(self):
        r = self.client.options(
            "/reset",
            headers={
                "Origin": "https://example.org",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-Session-Id,Content-Type",
            },
        )
        # Either 200 or 204; FastAPI's CORSMiddleware returns 200.
        self.assertIn(r.status_code, (200, 204))
        self.assertIn("access-control-allow-origin", {k.lower() for k in r.headers.keys()})


if __name__ == "__main__":
    unittest.main()
