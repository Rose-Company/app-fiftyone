from sqlalchemy.orm import Session
from app.models.character import Character
from app.schemas.character import CharacterCreate, CharacterUpdate
from app.repositories.base import BaseRepository
from typing import Optional, List
from uuid import UUID

class CharacterRepository(BaseRepository):
    def __init__(self, db: Session):
        super().__init__(Character, db)

    def get_by_video_id_and_code(self, video_id: UUID, character_code: str) -> Optional[Character]:
        return self.session.query(Character).filter(
            Character.video_id == video_id,
            Character.character_code == character_code
        ).first()

    def get_by_video_id(self, video_id: UUID) -> List[Character]:
        return self.session.query(Character).filter(Character.video_id == video_id).all()

    def create(self, character_data: CharacterCreate) -> Character:
        db_character = Character(**character_data.model_dump())
        self.session.add(db_character)
        self.session.commit()
        self.session.refresh(db_character)
        return db_character

    def update(self, character_id: UUID, character_data: CharacterUpdate) -> Optional[Character]:
        update_data = character_data.model_dump(exclude_unset=True)
        if not update_data:
            return None
        
        success = super().update_by_id(character_id, update_data)
        if success:
            return self.get_by_id(character_id)
        return None 