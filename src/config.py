"""Configuration loader."""
import os
import yaml
from pathlib import Path

_config = None

def load_config(path: str = None) -> dict:
    """Load config from YAML file."""
    global _config
    if _config is not None and path is None:
        return _config

    if path is None:
        # Look for config.yaml in project root
        root = Path(__file__).parent.parent
        path = root / "config.yaml"
        if not path.exists():
            path = root / "config.example.yaml"

    with open(path, "r") as f:
        _config = yaml.safe_load(f)

    # Allow env var overrides for secrets
    if os.environ.get("TOKOCRYPTO_API_KEY"):
        _config["exchange"]["api_key"] = os.environ["TOKOCRYPTO_API_KEY"]
    if os.environ.get("TOKOCRYPTO_SECRET"):
        _config["exchange"]["secret"] = os.environ["TOKOCRYPTO_SECRET"]
    if os.environ.get("CLAUDE_API_KEY"):
        _config["learning"]["claude_api_key"] = os.environ["CLAUDE_API_KEY"]
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        _config["notifications"]["telegram"]["bot_token"] = os.environ["TELEGRAM_BOT_TOKEN"]

    return _config


def get_db_path() -> str:
    """Get absolute path to database file."""
    cfg = load_config()
    db_path = Path(cfg["database"]["path"])
    if not db_path.is_absolute():
        db_path = Path(__file__).parent.parent / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return str(db_path)
