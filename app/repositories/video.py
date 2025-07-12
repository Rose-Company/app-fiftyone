from sqlalchemy.orm import Session
from app.models.video import Video
from app.schemas.video import VideoCreate, VideoUpdate
from app.repositories.base import BaseRepository
from typing import Optional
from uuid import UUID

class VideoRepository(BaseRepository):
    def __init__(self, db: Session):
        super().__init__(Video, db)

    def get_by_video_code(self, video_code: str) -> Optional[Video]:
        return self.session.query(Video).filter(Video.video_code == video_code).first()

    def create(self, video_data: VideoCreate) -> Video:
        db_video = Video(**video_data.model_dump())
        self.session.add(db_video)
        self.session.commit()
        self.session.refresh(db_video)
        return db_video

    def update(self, video_id: UUID, video_data: VideoUpdate) -> Optional[Video]:
        update_data = video_data.model_dump(exclude_unset=True)
        if not update_data:
            return None
        
        success = super().update_by_id(video_id, update_data)
        if success:
            result, error = self.get_by_id(video_id)
            return result if not error else None
        return None 