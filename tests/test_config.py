import importlib
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ConfigEnvVar(unittest.TestCase):
    def test_default_max_turns_is_5(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAX_TURNS", None)
            import config
            importlib.reload(config)
            self.assertEqual(config.MAX_TURNS, 5)

    def test_max_turns_env_override(self):
        with patch.dict(os.environ, {"MAX_TURNS": "8"}, clear=False):
            import config
            importlib.reload(config)
            self.assertEqual(config.MAX_TURNS, 8)

    def test_invalid_max_turns_raises(self):
        with patch.dict(os.environ, {"MAX_TURNS": "0"}, clear=False):
            import config
            with self.assertRaises(ValueError):
                importlib.reload(config)

    def tearDown(self):
        # Restore canonical default so other tests are not affected.
        os.environ.pop("MAX_TURNS", None)
        import config
        importlib.reload(config)


if __name__ == "__main__":
    unittest.main()
