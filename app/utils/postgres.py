from math import e
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get PostgreSQL configuration from environment variables
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASS = os.getenv('POSTGRES_PASSWORD', 'postgres')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'postgres')

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASS}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Debug logging
print(f"[DEBUG] DATABASE_URL: {DATABASE_URL}")
print(f"[DEBUG] POSTGRES_HOST: {POSTGRES_HOST}")
print(f"[DEBUG] POSTGRES_PORT: {POSTGRES_PORT}")
print(f"[DEBUG] POSTGRES_DB: {POSTGRES_DB}")
print(f"[DEBUG] POSTGRES_USER: {POSTGRES_USER}")
engine = create_engine(DATABASE_URL, echo=True, pool_size=5, max_overflow=10, pool_timeout=30, pool_recycle=1800, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Singleton Database Session Manager
class DatabaseManager:
    _instance = None
    _session = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance
    
    def get_session(self):
        if self._session is None:
            self._session = SessionLocal()
        return self._session
    
    def close_session(self):
        if self._session:
            self._session.close()
            self._session = None

# Global instance
db_manager = DatabaseManager()


def init_db():
    Base.metadata.create_all(bind=engine)


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
