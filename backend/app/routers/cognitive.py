from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.database import get_db
from app.models.models import User, CognitiveData
from app.schemas.schemas import CognitiveDataCreate, CognitiveDataResponse, PredictionResponse
from app.utils.auth import get_current_user
from app.services.prediction import predict_cognitive_load

router = APIRouter(prefix="/api/cognitive", tags=["Cognitive Data"])


@router.post("/log", response_model=CognitiveDataResponse)
def log_cognitive_data(
    data: CognitiveDataCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    features = [
        data.typing_speed,
        data.speed_variance,
        data.backspace_rate,
        data.mouse_distance,
        data.mouse_jitter,
        data.tab_switch_count,
    ]
    prediction = predict_cognitive_load(features)

    record = CognitiveData(
        user_id=current_user.id,
        typing_speed=data.typing_speed,
        speed_variance=data.speed_variance,
        backspace_rate=data.backspace_rate,
        mouse_distance=data.mouse_distance,
        mouse_jitter=data.mouse_jitter,
        tab_switch_count=data.tab_switch_count,
        predicted_load=prediction["predicted_load"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.post("/predict", response_model=PredictionResponse)
def predict(
    data: CognitiveDataCreate,
    current_user: User = Depends(get_current_user),
):
    features = [
        data.typing_speed,
        data.speed_variance,
        data.backspace_rate,
        data.mouse_distance,
        data.mouse_jitter,
        data.tab_switch_count,
    ]
    return predict_cognitive_load(features)


@router.get("/history", response_model=list[CognitiveDataResponse])
def get_history(
    days: int = Query(default=7, ge=1, le=90),
    limit: int = Query(default=500, ge=1, le=5000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    return (
        db.query(CognitiveData)
        .filter(
            CognitiveData.user_id == current_user.id,
            CognitiveData.timestamp >= since,
        )
        .order_by(CognitiveData.timestamp.desc())
        .limit(limit)
        .all()
    )


@router.get("/latest", response_model=Optional[CognitiveDataResponse])
def get_latest(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(CognitiveData)
        .filter(CognitiveData.user_id == current_user.id)
        .order_by(CognitiveData.timestamp.desc())
        .first()
    )


@router.get("/stats")
def get_stats(
    days: int = Query(default=7, ge=1, le=90),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    records = (
        db.query(CognitiveData)
        .filter(
            CognitiveData.user_id == current_user.id,
            CognitiveData.timestamp >= since,
        )
        .order_by(CognitiveData.timestamp.asc())
        .all()
    )

    if not records:
        return {
            "total_records": 0,
            "high_load_count": 0,
            "medium_load_count": 0,
            "low_load_count": 0,
            "avg_typing_speed": 0,
            "avg_backspace_rate": 0,
            "avg_mouse_jitter": 0,
            "high_load_streak_minutes": 0,
        }

    high_count = sum(1 for r in records if r.predicted_load == "High")
    medium_count = sum(1 for r in records if r.predicted_load == "Medium")
    low_count = sum(1 for r in records if r.predicted_load == "Low")
    avg_typing = sum(r.typing_speed for r in records) / len(records)
    avg_backspace = sum(r.backspace_rate for r in records) / len(records)
    avg_jitter = sum(r.mouse_jitter for r in records) / len(records)

    # Calculate longest high-load streak in minutes
    max_streak = 0
    current_streak_start = None
    for r in records:
        if r.predicted_load == "High":
            if current_streak_start is None:
                current_streak_start = r.timestamp
        else:
            if current_streak_start is not None:
                delta = (r.timestamp - current_streak_start).total_seconds() / 60
                max_streak = max(max_streak, delta)
                current_streak_start = None
    if current_streak_start and records:
        delta = (records[-1].timestamp - current_streak_start).total_seconds() / 60
        max_streak = max(max_streak, delta)

    return {
        "total_records": len(records),
        "high_load_count": high_count,
        "medium_load_count": medium_count,
        "low_load_count": low_count,
        "avg_typing_speed": round(avg_typing, 2),
        "avg_backspace_rate": round(avg_backspace, 2),
        "avg_mouse_jitter": round(avg_jitter, 2),
        "high_load_streak_minutes": round(max_streak, 1),
    }
