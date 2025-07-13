from sqlalchemy import Column, String, Boolean, Integer, DateTime, Numeric, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.utils.postgres import Base
import uuid

class CharacterAppearance(Base):
    __tablename__ = "character_appearances"

    id = Column(Integer, primary_key=True)
    character_id = Column(UUID(as_uuid=True), ForeignKey("characters.id"), nullable=False)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"), nullable=False)
    start_time = Column(Numeric(10, 3), nullable=False)
    end_time = Column(Numeric(10, 3), nullable=False)
    duration = Column(Numeric(10, 3), nullable=True, default=0)
    confidence = Column(Numeric(3, 2), nullable=True, default=0.0)
    is_confirmed = Column(Boolean, nullable=True, default=False)
    color_shown = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_by = Column(UUID(as_uuid=True), nullable=True)
    name_videoid = Column(String, nullable=True) 