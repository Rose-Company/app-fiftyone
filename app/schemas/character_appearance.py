from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from uuid import UUID
from decimal import Decimal

class CharacterAppearanceBase(BaseModel):
    character_id: UUID
    video_id: UUID
    start_time: Decimal
    end_time: Decimal
    duration: Optional[Decimal] = Decimal('0')
    confidence: Optional[Decimal] = Decimal('0')
    is_confirmed: Optional[bool] = False
    name_videoid: Optional[str] = None

class CharacterAppearanceCreate(CharacterAppearanceBase):
    pass

class CharacterAppearanceUpdate(BaseModel):
    is_confirmed: Optional[bool] = None
    confidence: Optional[Decimal] = None

class CharacterAppearanceInDB(CharacterAppearanceBase):
    id: int
    created_at: datetime
    created_by: Optional[UUID] = None
    color_shown: Optional[str] = None

    class Config:
        from_attributes = True 