from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from uuid import UUID

class VideoBase(BaseModel):
    title: str
    file_path: str
    duration: int
    video_code: str

class VideoCreate(VideoBase):
    pass

class VideoUpdate(BaseModel):
    has_character_analysis: Optional[bool] = None
    character_count: Optional[int] = None
    status: Optional[str] = None

class VideoInDB(VideoBase):
    id: UUID
    status: str
    has_character_analysis: bool
    character_count: int
    created_at: datetime
    created_by: Optional[UUID] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[UUID] = None

    class Config:
        from_attributes = True 