from sqlalchemy import Column, String, Boolean, Integer, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.utils.postgres import Base
import uuid

class Video(Base):
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String, nullable=False, default="pending")
    title = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    duration = Column(Integer, nullable=False)
    thumbnail_url = Column(String, nullable=True)
    has_character_analysis = Column(Boolean, nullable=True, default=False)
    character_count = Column(Integer, nullable=True, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_by = Column(UUID(as_uuid=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=True)
    updated_by = Column(UUID(as_uuid=True), nullable=True)
    metadata = Column(JSON, nullable=True)
    video_code = Column(String, nullable=True) 