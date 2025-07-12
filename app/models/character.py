from sqlalchemy import Column, String, Boolean, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.utils.postgres import Base
import uuid

class Character(Base):
    __tablename__ = "characters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    avatar = Column(String, nullable=True)
    is_active = Column(Boolean, nullable=True, default=True)
    gender = Column(String, nullable=True)
    character_type = Column(String, nullable=True, default="person")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_by = Column(UUID(as_uuid=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=True)
    updated_by = Column(UUID(as_uuid=True), nullable=True)
    metadata = Column(JSON, nullable=True)
    video_id = Column(UUID(as_uuid=True), nullable=True)
    character_code = Column(String, nullable=True) 