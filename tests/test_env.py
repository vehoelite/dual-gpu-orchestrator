from orchestrator.env import load_dotenv


def test_parses_keys_comments_quotes_blanks(tmp_path):
    f = tmp_path / ".env"
    f.write_text(
        "# a comment\n"
        "\n"
        'GEMINI_API_KEY="abc123"\n'
        "LMSTUDIO_TOKEN = sk-lm-xyz \n"
        "NOT_A_PAIR\n",
        encoding="utf-8",
    )
    env = {}
    loaded = load_dotenv(f, environ=env)
    assert env["GEMINI_API_KEY"] == "abc123"
    assert env["LMSTUDIO_TOKEN"] == "sk-lm-xyz"
    assert "NOT_A_PAIR" not in env
    assert loaded == {"GEMINI_API_KEY": "abc123", "LMSTUDIO_TOKEN": "sk-lm-xyz"}


def test_does_not_override_existing(tmp_path):
    f = tmp_path / ".env"
    f.write_text("LMSTUDIO_TOKEN=from-file\n", encoding="utf-8")
    env = {"LMSTUDIO_TOKEN": "from-real-env"}
    load_dotenv(f, environ=env)
    assert env["LMSTUDIO_TOKEN"] == "from-real-env"  # real env wins


def test_missing_file_is_noop(tmp_path):
    assert load_dotenv(tmp_path / "nope.env", environ={}) == {}
