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
