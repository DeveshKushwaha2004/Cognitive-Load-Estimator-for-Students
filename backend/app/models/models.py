from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    cognitive_data = relationship("CognitiveData", back_populates="user")


class CognitiveData(Base):
    __tablename__ = "cognitive_data"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    typing_speed = Column(Float, default=0.0)
    speed_variance = Column(Float, default=0.0)
    backspace_rate = Column(Float, default=0.0)
    mouse_distance = Column(Float, default=0.0)
    mouse_jitter = Column(Float, default=0.0)
    tab_switch_count = Column(Float, default=0.0)
    predicted_load = Column(String, default="Low")

    user = relationship("User", back_populates="cognitive_data")
