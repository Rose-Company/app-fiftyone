import signal
import sys
from fastapi import APIRouter
from app.controllers.migration_controller import MigrationController
from app.utils.logger import get_logger

logger = get_logger(__name__)

def signal_handler(signum, frame):
    logger.info("Received shutdown signal")
    sys.exit(0)

def init_migration_controller():
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Create and run migration controller
        controller = MigrationController()
        logger.info("Starting migration service...")
        return controller
        
    except Exception as e:
        logger.error(f"Error initializing migration controller: {str(e)}")
        raise

def create_migrate_controllers(repo) -> APIRouter:
    router = APIRouter(tags=["migration"])
    
    @router.get("/migration/status")
    async def get_migration_status():
        return {"status": "running"}
    
    return router

def create_quiz_controllers(repo) -> APIRouter:
    router = APIRouter(tags=["quiz"])
    
    @router.get("/quiz/status")
    async def get_quiz_status():
        return {"status": "running"}
    
    return router