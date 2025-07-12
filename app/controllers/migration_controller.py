import os
from app.services.migration_service import MigrationService
from app.utils.postgres import db_session
from app.utils.logger import get_logger

logger = get_logger(__name__)

class MigrationController:
    def __init__(self, result_dir: str = "app/data/result"):
        self.result_dir = result_dir
        self.processed_files = set()

    def run(self):
        """
        Single run to scan directory and process files
        """
        logger.info("Running migration controller...")
        logger.info(f"Checking directory: {self.result_dir}")
        logger.info(f"Directory exists: {os.path.exists(self.result_dir)}")
        
        try:
            # Use session-per-operation with proper transaction management
            with db_session() as db:
                migration_service = MigrationService(db)
                
                # Scan directory for new files
                if os.path.exists(self.result_dir):
                    all_files = os.listdir(self.result_dir)
                    logger.info(f"Files in directory: {all_files}")
                    
                    current_files = set(f for f in all_files if f.endswith('.json'))
                    logger.info(f"JSON files found: {current_files}")
                    logger.info(f"Previously processed: {self.processed_files}")
                    
                    new_files = current_files - self.processed_files
                    logger.info(f"New files to process: {new_files}")
                    
                    if new_files:
                        logger.info(f"Found {len(new_files)} new files to process: {list(new_files)}")
                        
                        # Process each new file
                        results = migration_service.scan_and_process_directory(self.result_dir)
                        
                        # Log results
                        if results["success"]:
                            logger.info(f"Successfully processed files: {results['success']}")
                            self.processed_files.update(results["success"])
                        
                        if results["failed"]:
                            logger.error(f"Failed to process files: {results['failed']}")
                    
                    else:
                        logger.info("No new files found to process")
                
                else:
                    logger.warning(f"Result directory {self.result_dir} does not exist")
                    logger.info(f"Current working directory: {os.getcwd()}")
                    # Try to list what directories exist
                    try:
                        parent_dir = os.path.dirname(self.result_dir)
                        if os.path.exists(parent_dir):
                            logger.info(f"Parent directory contents: {os.listdir(parent_dir)}")
                    except Exception as e:
                        logger.error(f"Could not list parent directory: {e}")
            
        except Exception as e:
            logger.error(f"Error in migration controller: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def stop(self):
        """
        Stop the controller (placeholder for cleanup if needed)
        """
        logger.info("Stopping migration controller...") 