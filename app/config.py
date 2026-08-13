from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    face_api_keys: str = ""
    face_model_name: str = "buffalo_l"
    face_providers: str = "cpu"

    face_match_threshold: float = 0.40
    face_min_det_score: float = 0.50
    face_min_face_ratio: float = 5.0
    face_min_laplacian_var: float = 50.0
    face_min_brightness: float = 40.0
    face_max_brightness: float = 220.0

    face_max_image_mb: float = 10.0
    face_min_resolution: int = 200
    face_models_root: str = "/app/models"

    port: int = 8000
    app_version: str = "1.0.0"

    @property
    def api_keys(self) -> set[str]:
        return {key.strip() for key in self.face_api_keys.split(",") if key.strip()}

    @property
    def onnx_providers(self) -> List[str]:
        if self.face_providers.lower() == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]


settings = Settings()
