"""
Cloud-ready database configuration for deployment
Supports both local and cloud PostgreSQL databases
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .database import Base, DatabaseManager
import logging

logger = logging.getLogger(__name__)

class CloudDatabaseManager(DatabaseManager):
    """Cloud-ready database manager with environment variable support"""
    
    def __init__(self):
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Try to get database URL from environment (for cloud deployment)
        database_url = os.getenv('DATABASE_URL')
        
        if database_url:
            # Cloud deployment - use DATABASE_URL
            logger.info("Using cloud database configuration")
            self.connection_string = database_url
        else:
            # Local development - use default PostgreSQL settings
            logger.info("Using local database configuration")
            host = os.getenv('DB_HOST', '127.0.0.1')
            port = os.getenv('DB_PORT', '5432')
            dbname = os.getenv('DB_NAME', 'postgres')
            user = os.getenv('DB_USER', 'postgres')
            password = os.getenv('DB_PASSWORD', '1234')
            
            self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        
        self.engine = None
        self.Session = None
        self.setup_database()
    
    def setup_database(self):
        """Set up database connection and create tables"""
        try:
            # Handle SSL requirements for cloud databases
            if 'DATABASE_URL' in os.environ:
                # Cloud database - may require SSL
                self.engine = create_engine(
                    self.connection_string,
                    connect_args={"sslmode": "require"} if 'heroku' in self.connection_string else {}
                )
            else:
                # Local database
                self.engine = create_engine(self.connection_string)
            
            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up database: {str(e)}")
            raise

def get_database_manager():
    """Factory function to get appropriate database manager"""
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check if we're in a cloud environment (Streamlit Cloud, Heroku, etc.)
        # In cloud environments, we should always use Supabase
        is_cloud_env = (
            os.getenv('STREAMLIT_SHARING_MODE') or  # Streamlit Cloud
            os.getenv('DYNO') or  # Heroku
            os.getenv('RAILWAY_ENVIRONMENT') or  # Railway
            os.getenv('RENDER') or  # Render
            os.getenv('VERCEL') or  # Vercel
            os.getenv('DATABASE_URL')  # Any cloud platform with DATABASE_URL
        )
        
        if is_cloud_env:
            logger.info("Cloud environment detected, using Supabase database")
            from .supabase_database import get_supabase_manager
            return get_supabase_manager()
        else:
            # Local development - check for DATABASE_URL first
            database_url = os.getenv('DATABASE_URL')
            if database_url and 'supabase.co' in database_url:
                logger.info("Using Supabase database (local with Supabase)")
                from .supabase_database import get_supabase_manager
                return get_supabase_manager()
            else:
                logger.info("Using local database configuration")
                return CloudDatabaseManager()
    except Exception as e:
        logger.warning(f"Cloud database not available, falling back to local: {str(e)}")
        from .database import DatabaseManager
        return DatabaseManager()  # Fallback to local
