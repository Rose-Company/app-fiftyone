#!/usr/bin/env python3
"""
Main Flask application cho migration toolkit
"""
import os
import signal
import sys
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import uvicorn
from app.utils.postgres import init_db, db_session
from sqlalchemy import text
from app.controllers import init_migration_controller
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(title="Migration Service API")
scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def startup_event():
    try:
        # Initialize database
        init_db()
        
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
    try:
        # Check database connection using db_session
        with db_session() as db:
            db.execute(text("SELECT 1"))
            return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 