#!/usr/bin/env python3
"""
Main Flask application cho migration toolkit
"""
import os
import sys
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from factory import create_app
from utils.postgres import init_db
from controllers import init_migration_controller
from utils.logger import get_logger

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(title="Migration Service API")
scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def startup_event():
    try:
        # Initialize Flask app and database
        flask_app = create_app()
        init_db(flask_app)
        
        # Initialize migration controller
        controller = init_migration_controller()
        
        # Add job to scheduler
        scheduler.add_job(
            controller.run,
            trigger=IntervalTrigger(seconds=10),
            id="migration_job",
            name="Migration Job",
            replace_existing=True
        )
        
        # Start the scheduler
        scheduler.start()
        logger.info("Migration service scheduler started")
        
    except Exception as e:
        logger.error(f"Error in startup: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
    logger.info("Migration service scheduler shutdown")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/jobs")
async def list_jobs():
    jobs = scheduler.get_jobs()
    return {
        "jobs": [
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": str(job.next_run_time)
            }
            for job in jobs
        ]
    }

# This is important - define the app variable at module level
api = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:api",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 