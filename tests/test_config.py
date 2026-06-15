from orchestrator.config import Config


def test_defaults():
    cfg = Config()
    assert cfg.lm_studio_url == "http://localhost:1234/v1"
    assert cfg.max_steps > 0
    assert cfg.command_timeout > 0
    assert cfg.request_timeout > 0


def test_override():
    cfg = Config(lm_studio_url="http://x/v1", max_steps=5)
    assert cfg.lm_studio_url == "http://x/v1"
    assert cfg.max_steps == 5


def test_phase2_defaults():
    cfg = Config()
    assert cfg.planner == "local"
    assert cfg.planner_fallback_local is True
    assert cfg.max_dominant_turns > 0
    assert cfg.no_progress_limit > 0
    assert cfg.gemini_model
