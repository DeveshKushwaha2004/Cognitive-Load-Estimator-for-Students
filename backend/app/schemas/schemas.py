from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


# --- Auth Schemas ---
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    name: str
    email: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    user_id: Optional[int] = None


# --- Cognitive Data Schemas ---
class CognitiveDataCreate(BaseModel):
    typing_speed: float
    speed_variance: float
    backspace_rate: float
    mouse_distance: float
    mouse_jitter: float
    tab_switch_count: float


class CognitiveDataResponse(BaseModel):
    id: int
    user_id: int
    timestamp: datetime
    typing_speed: float
    speed_variance: float
    backspace_rate: float
    mouse_distance: float
    mouse_jitter: float
    tab_switch_count: float
    predicted_load: str

    class Config:
        from_attributes = True


class PredictionResponse(BaseModel):
    predicted_load: str
    confidence: float
    load_percentage: float
