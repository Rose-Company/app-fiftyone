import os
from app.services.migration_service import MigrationService
from app.utils.postgres import db_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)

class MigrationController:
    def __init__(self, result_dir: str = "data/result"):
        self.result_dir = result_dir
        self.processed_files = set()

    def run(self):
        """
        Single run to scan directory and process files
        """
        logger.info("Running migration controller...")
        
        try:
            # Get singleton database session
            db = db_manager.get_session()
            migration_service = MigrationService(db)
            
            # Scan directory for new files
            if os.path.exists(self.result_dir):
                current_files = set(f for f in os.listdir(self.result_dir) 
                                 if f.endswith('.json'))
                new_files = current_files - self.processed_files
                
                if new_files:
                    logger.info(f"Found {len(new_files)} new files to process")
                    
                    # Process each new file
                    results = migration_service.scan_and_process_directory(self.result_dir)
                    
                    # Log results
                    if results["success"]:
                        logger.info(f"Successfully processed files: {results['success']}")
                        self.processed_files.update(results["success"])
                    
                    if results["failed"]:
                        logger.error(f"Failed to process files: {results['failed']}")
                
                else:
                    logger.debug("No new files found")
            
            else:
                logger.warning(f"Result directory {self.result_dir} does not exist")
            
        except Exception as e:
            logger.error(f"Error in migration controller: {str(e)}")

    def stop(self):
        """
        Stop the controller (placeholder for cleanup if needed)
        """
        logger.info("Stopping migration controller...") 