"""Application settings placeholder."""

from pydantic import BaseModel


class AppSettings(BaseModel):
    project_name: str = "graphrag-pipeline"
    models_config_path: str = "configs/models.yaml"
    pipeline_config_path: str = "configs/pipeline.yaml"
    evaluation_config_path: str = "configs/evaluation.yaml"
