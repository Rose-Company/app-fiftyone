from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from uuid import UUID

class CharacterBase(BaseModel):
    name: str
    video_id: UUID
    character_code: str
    character_type: Optional[str] = "person"
    is_active: Optional[bool] = True

class CharacterCreate(CharacterBase):
    pass

class CharacterUpdate(BaseModel):
    is_active: Optional[bool] = None
    gender: Optional[str] = None
    character_type: Optional[str] = None

class CharacterInDB(CharacterBase):
    id: UUID
    avatar: Optional[str] = None
    gender: Optional[str] = None
    created_at: datetime
    created_by: Optional[UUID] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[UUID] = None

    class Config:
        from_attributes = True 