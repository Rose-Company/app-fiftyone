
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_logger(name: str = __name__) -> logging.Logger:
    return logging.getLogger(name)

# For backward compatibility
logger = get_logger(__name__)