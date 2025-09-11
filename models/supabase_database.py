"""
Supabase-specific database configuration
Handles Supabase PostgreSQL connection and table creation
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from .database import Base, DatabaseManager
import logging

logger = logging.getLogger(__name__)

class SupabaseDatabaseManager(DatabaseManager):
    """Supabase-specific database manager"""
    
    def __init__(self):
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Get Supabase connection details from environment variables
        database_url = os.getenv('DATABASE_URL')
        if database_url and 'supabase.co' in database_url:
            # Use DATABASE_URL if it's a Supabase URL
            self.connection_string = database_url
        else:
            # Fallback to individual environment variables
            host = os.getenv('SUPABASE_HOST', 'db.nxnteidyxmznwfdupoav.supabase.co')
            port = os.getenv('SUPABASE_PORT', '5432')
            database = os.getenv('SUPABASE_DB', 'postgres')
            user = os.getenv('SUPABASE_USER', 'postgres')
            password = os.getenv('SUPABASE_PASSWORD', '1234')
            
            self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        self.engine = None
        self.Session = None
        self.setup_database()
    
    def setup_database(self):
        """Set up Supabase database connection and create tables"""
        try:
            # Create engine with Supabase-specific settings
            self.engine = create_engine(
                self.connection_string,
                connect_args={
                    "sslmode": "require",  # Supabase requires SSL
                    "connect_timeout": 10,
                    "application_name": "medical_ai_assistant"
                },
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=300     # Recycle connections every 5 minutes
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                logger.info(f"Connected to Supabase PostgreSQL: {version}")
            
            # Create all tables
            Base.metadata.create_all(self.engine)
            logger.info("All tables created successfully in Supabase")
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            # Verify tables exist
            self.verify_tables()
            
            logger.info("Supabase database setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up Supabase database: {str(e)}")
            raise
    
    def verify_tables(self):
        """Verify that all required tables exist in Supabase"""
        try:
            with self.engine.connect() as conn:
                # Check if tables exist
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('documents', 'document_chunks', 'chat_messages', 'message_sources', 
                                      'medical_reports', 'medical_tests', 'medical_reports_wide')
                """))
                
                existing_tables = [row[0] for row in result.fetchall()]
                required_tables = ['documents', 'document_chunks', 'chat_messages', 'message_sources', 
                                 'medical_reports', 'medical_tests', 'medical_reports_wide']
                
                missing_tables = set(required_tables) - set(existing_tables)
                
                if missing_tables:
                    logger.warning(f"Missing tables in Supabase: {missing_tables}")
                    # Try to create missing tables
                    Base.metadata.create_all(self.engine)
                    logger.info("Attempted to create missing tables")
                else:
                    logger.info("All required tables exist in Supabase")
                    
        except Exception as e:
            logger.error(f"Error verifying tables: {str(e)}")
            raise
    
    def test_connection(self):
        """Test Supabase connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                test_value = result.fetchone()[0]
                if test_value == 1:
                    logger.info("✅ Supabase connection test successful")
                    return True
                else:
                    logger.error("❌ Supabase connection test failed")
                    return False
        except Exception as e:
            logger.error(f"❌ Supabase connection test failed: {str(e)}")
            return False
    
    def get_connection_info(self):
        """Get connection information for debugging"""
        # Parse connection string to extract info
        if 'supabase.co' in self.connection_string:
            host = 'db.nxnteidyxmznwfdupoav.supabase.co'
            port = 5432
            database = 'postgres'
            user = 'postgres'
        else:
            # Parse from connection string
            import re
            match = re.match(r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', self.connection_string)
            if match:
                user, password, host, port, database = match.groups()
                port = int(port)
            else:
                host = port = database = user = "unknown"
        
        return {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "ssl": "required",
            "status": "connected" if self.engine else "disconnected",
            "connection_string": self.connection_string[:50] + "..." if len(self.connection_string) > 50 else self.connection_string
        }

def get_supabase_manager():
    """Factory function to get Supabase database manager"""
    try:
        return SupabaseDatabaseManager()
    except Exception as e:
        logger.error(f"Failed to create Supabase manager: {str(e)}")
        raise
