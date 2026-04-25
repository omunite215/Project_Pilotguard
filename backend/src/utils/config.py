"""PilotGuard application configuration via Pydantic Settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""

    # Paths
    project_root: Path = Path(__file__).resolve().parents[3]
    config_path: Path = Path(__file__).resolve().parents[3] / "configs" / "training_config.yaml"
    models_dir: Path = Path(__file__).resolve().parents[2] / "models"
    data_dir: Path = Path(__file__).resolve().parents[3] / "data"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # CV Pipeline
    landmark_backend: str = "mediapipe"
    frame_width: int = 640
    frame_height: int = 480

    # Thresholds
    ear_threshold_multiplier: float = 0.75
    calibration_duration_seconds: int = 30
    perclos_window_seconds: int = 60
    microsleep_threshold_ms: int = 500

    # Alerts
    advisory_threshold: float = 30.0
    caution_threshold: float = 55.0
    warning_threshold: float = 75.0
    alert_cooldown_seconds: int = 30

    # LLM
    llm_api_key: str = ""
    llm_rate_limit_seconds: int = 60

    model_config = {"env_prefix": "PILOTGUARD_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
