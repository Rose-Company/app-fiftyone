from typing import Optional
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from app.controllers import (
    create_migrate_controllers,
    create_quiz_controllers,
)

from app.repositories import (
    VideoRepository,
    CharacterRepository,
    CharacterAppearanceRepository,
)

from app.utils import (
    init_db, 
    const
)

from app.utils.postgres import db_session

def create_app() -> FastAPI:
    app = FastAPI(title="Migration Service API")

    # Setup CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize database
    init_db()

    # Create repositories and controllers using session-per-request
    with db_session() as session:
        video_repo = VideoRepository(session)
        character_repo = CharacterRepository(session)
        character_appearance_repo = CharacterAppearanceRepository(session)

        # Initialize routers
        migrate_router = create_migrate_controllers(video_repo)
        quiz_router = create_quiz_controllers(character_repo)

        # Include routers with prefix
        app.include_router(migrate_router, prefix=const.URL_PREFIX)
        app.include_router(quiz_router, prefix=const.URL_PREFIX)

    return app
