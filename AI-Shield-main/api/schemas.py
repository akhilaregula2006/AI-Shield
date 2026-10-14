"""Pydantic request/response models."""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.ingestion.loader import V_COLUMNS


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TransactionRequest(BaseModel):
    time: float = Field(ge=0, description="Seconds elapsed since first transaction in the dataset")
    amount: float = Field(ge=0)
    features: Dict[str, float] = Field(description="PCA components V1..V28")
    explain: bool = False

    @field_validator("features")
    @classmethod
    def _require_all_components(cls, value: Dict[str, float]) -> Dict[str, float]:
        missing = [c for c in V_COLUMNS if c not in value]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return value


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_version: int
    explanation: Optional[List[FeatureContribution]] = None


class FeedbackRequest(TransactionRequest):
    is_fraud: bool


class CopilotRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
