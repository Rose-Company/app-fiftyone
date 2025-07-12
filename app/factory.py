from typing import Optional
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from app.controllers import (
    create_migrate_controllers,
    create_quiz_controllers,
)

from app.repositories import (
    MigrateRepository,
    QuizRepository,
)

from app.utils import (
    init_db, 
    db_session,
    const
)

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
    init_db(app)

    # Create repositories and controllers
    with db_session() as session:
        migrate_repo = MigrateRepository(session)
        quiz_repo = QuizRepository(session)

        # Initialize routers
        migrate_router = create_migrate_controllers(migrate_repo)
        quiz_router = create_quiz_controllers(quiz_repo)

        # Include routers with prefix
        app.include_router(migrate_router, prefix=const.URL_PREFIX)
        app.include_router(quiz_router, prefix=const.URL_PREFIX)

    return app
