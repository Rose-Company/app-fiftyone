from sqlalchemy.orm import Session
from sqlalchemy import cast, String
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from app.models.character_appearance import CharacterAppearance
from app.schemas.character_appearance import CharacterAppearanceCreate, CharacterAppearanceUpdate
from app.repositories.base import BaseRepository
from typing import Optional, List
from uuid import UUID

class CharacterAppearanceRepository(BaseRepository):
    def __init__(self, db: Session):
        super().__init__(CharacterAppearance, db)

    def get_by_character_and_video(self, character_id: UUID, video_id: UUID) -> List[CharacterAppearance]:
        # Ensure UUIDs are proper UUID objects
        if not isinstance(character_id, UUID):
            character_id = UUID(str(character_id))
        if not isinstance(video_id, UUID):
            video_id = UUID(str(video_id))
            
        return self.session.query(CharacterAppearance).filter(
            cast(CharacterAppearance.character_id, String) == str(character_id),
            cast(CharacterAppearance.video_id, String) == str(video_id)
        ).all()

    def create(self, appearance_data: CharacterAppearanceCreate) -> CharacterAppearance:
        db_appearance = CharacterAppearance(**appearance_data.model_dump())
        self.session.add(db_appearance)
        self.session.commit()
        self.session.refresh(db_appearance)
        return db_appearance

    def create_many(self, appearances_data: List[CharacterAppearanceCreate]) -> List[CharacterAppearance]:
        db_appearances = [CharacterAppearance(**data.model_dump()) for data in appearances_data]
        self.session.add_all(db_appearances)
        self.session.commit()
        for appearance in db_appearances:
            self.session.refresh(appearance)
        return db_appearances

    def update(self, appearance_id: int, appearance_data: CharacterAppearanceUpdate) -> Optional[CharacterAppearance]:
        update_data = appearance_data.model_dump(exclude_unset=True)
        if not update_data:
            return None
        
        success = super().update_by_id(appearance_id, update_data)
        if success:
            result, error = self.get_by_id(appearance_id)
            return result if not error else None
        return None 