"""
Reusable database operation utilities for the Medical AI Assistant
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from utils.error_handler import ErrorHandler, retry_on_failure, DatabaseError

logger = logging.getLogger(__name__)

class DatabaseConnectionManager:
    """Manages database connections with proper error handling"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.Session = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection"""
        try:
            self.engine = create_engine(
                self.database_url,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False
            )
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database connection initialized successfully")
        except Exception as e:
            error_id = ErrorHandler.log_error(e, "database_connection_init")
            raise DatabaseError(f"Failed to initialize database connection (Error ID: {error_id})")
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def get_session(self):
        """Get a new database session"""
        try:
            return self.Session()
        except Exception as e:
            error_id = ErrorHandler.log_error(e, "get_database_session")
            raise DatabaseError(f"Failed to create database session (Error ID: {error_id})")
    
    def close_connection(self):
        """Close database connection"""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Database connection closed")
        except Exception as e:
            ErrorHandler.log_error(e, "close_database_connection")

class QueryExecutor:
    """Executes database queries with error handling and retry logic"""
    
    def __init__(self, connection_manager: DatabaseConnectionManager):
        self.connection_manager = connection_manager
    
    @retry_on_failure(max_retries=2, delay=0.5)
    def execute_query(
        self,
        query: str,
        params: Dict[str, Any] = None,
        fetch_one: bool = False,
        fetch_all: bool = True,
        context: str = "database_query"
    ) -> Tuple[bool, Optional[Union[List[Dict], Dict]], Optional[str]]:
        """
        Execute a database query with error handling
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch_one: Return single result
            fetch_all: Return all results
            context: Context for error logging
            
        Returns:
            Tuple of (success, results, error_message)
        """
        session = None
        try:
            session = self.connection_manager.get_session()
            
            # Execute query
            result = session.execute(text(query), params or {})
            
            if fetch_one:
                row = result.fetchone()
                if row:
                    return True, dict(row._mapping), None
                else:
                    return True, None, None
            elif fetch_all:
                rows = result.fetchall()
                return True, [dict(row._mapping) for row in rows], None
            else:
                return True, None, None
                
        except SQLAlchemyError as e:
            error_id = ErrorHandler.log_error(e, context, {
                'query': query[:200],  # Truncate long queries
                'params': str(params)[:200]
            })
            return False, None, f"Database error (ID: {error_id}): {str(e)}"
        except Exception as e:
            error_id = ErrorHandler.log_error(e, context, {
                'query': query[:200],
                'params': str(params)[:200]
            })
            return False, None, f"Unexpected error (ID: {error_id}): {str(e)}"
        finally:
            if session:
                session.close()
    
    def execute_insert(
        self,
        query: str,
        params: Dict[str, Any] = None,
        context: str = "database_insert"
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Execute an INSERT query and return the inserted ID
        
        Args:
            query: SQL INSERT query
            params: Query parameters
            context: Context for error logging
            
        Returns:
            Tuple of (success, inserted_id, error_message)
        """
        session = None
        try:
            session = self.connection_manager.get_session()
            
            result = session.execute(text(query), params or {})
            session.commit()
            
            # Get the inserted ID
            inserted_id = result.lastrowid
            
            return True, inserted_id, None
            
        except SQLAlchemyError as e:
            if session:
                session.rollback()
            error_id = ErrorHandler.log_error(e, context, {
                'query': query[:200],
                'params': str(params)[:200]
            })
            return False, None, f"Database insert error (ID: {error_id}): {str(e)}"
        except Exception as e:
            if session:
                session.rollback()
            error_id = ErrorHandler.log_error(e, context, {
                'query': query[:200],
                'params': str(params)[:200]
            })
            return False, None, f"Unexpected error (ID: {error_id}): {str(e)}"
        finally:
            if session:
                session.close()

class MedicalDataProcessor:
    """Processes medical data with validation and normalization"""
    
    @staticmethod
    def validate_medical_value(test_name: str, value: Union[str, float], unit: str = "") -> Tuple[bool, str]:
        """
        Validate medical test value
        
        Args:
            test_name: Name of the medical test
            value: Test value
            unit: Test unit
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        try:
            # Convert to float if possible
            if isinstance(value, str):
                # Extract numeric value from string
                import re
                numeric_match = re.search(r'[\d.]+', value)
                if not numeric_match:
                    return False, f"Could not extract numeric value from '{value}'"
                value = float(numeric_match.group())
            
            # Check if value is within reasonable range
            if value < 0:
                return False, "Value cannot be negative"
            
            if value > 10000:  # Arbitrary upper limit
                return False, "Value appears to be unreasonably high"
            
            return True, "Valid medical value"
            
        except (ValueError, TypeError) as e:
            return False, f"Invalid value format: {str(e)}"
    
    @staticmethod
    def normalize_test_name(test_name: str) -> str:
        """
        Normalize medical test name for consistent storage
        
        Args:
            test_name: Raw test name
            
        Returns:
            Normalized test name
        """
        if not test_name:
            return ""
        
        # Convert to lowercase and replace spaces with underscores
        normalized = test_name.lower().strip()
        normalized = normalized.replace(' ', '_')
        normalized = normalized.replace('-', '_')
        normalized = normalized.replace('(', '').replace(')', '')
        
        return normalized
    
    @staticmethod
    def categorize_test_type(test_name: str) -> str:
        """
        Categorize medical test by type
        
        Args:
            test_name: Medical test name
            
        Returns:
            Test category
        """
        test_lower = test_name.lower()
        
        # Biochemistry tests
        if any(term in test_lower for term in ['glucose', 'sugar', 'rbs', 'fbs', 'hba1c']):
            return 'biochemistry'
        elif any(term in test_lower for term in ['cholesterol', 'triglyceride', 'hdl', 'ldl', 'vldl']):
            return 'lipid'
        elif any(term in test_lower for term in ['creatinine', 'urea', 'bun', 'uric', 'phosphorus']):
            return 'kidney'
        elif any(term in test_lower for term in ['sgot', 'sgpt', 'ast', 'alt', 'ggt', 'bilirubin', 'alkaline']):
            return 'liver'
        elif any(term in test_lower for term in ['tsh', 't3', 't4', 'thyroid']):
            return 'thyroid'
        elif any(term in test_lower for term in ['vitamin', 'calcium', 'sodium', 'potassium']):
            return 'vitamins_minerals'
        else:
            return 'other'

class ConversationManager:
    """Manages conversation data and history"""
    
    def __init__(self, query_executor: QueryExecutor):
        self.query_executor = query_executor
    
    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Save a chat message to the database
        
        Args:
            conversation_id: Unique conversation identifier
            role: Message role (user/assistant)
            content: Message content
            metadata: Additional message metadata
            
        Returns:
            Tuple of (success, message_id, error_message)
        """
        query = """
        INSERT INTO chat_messages (conversation_id, role, content, metadata, created_at)
        VALUES (:conversation_id, :role, :content, :metadata, datetime('now'))
        """
        
        params = {
            'conversation_id': conversation_id,
            'role': role,
            'content': content,
            'metadata': str(metadata) if metadata else None
        }
        
        return self.query_executor.execute_insert(
            query, params, f"save_message_{role}"
        )
    
    def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 50
    ) -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
        """
        Get conversation history
        
        Args:
            conversation_id: Unique conversation identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            Tuple of (success, messages, error_message)
        """
        query = """
        SELECT role, content, metadata, created_at
        FROM chat_messages
        WHERE conversation_id = :conversation_id
        ORDER BY created_at ASC
        LIMIT :limit
        """
        
        params = {
            'conversation_id': conversation_id,
            'limit': limit
        }
        
        return self.query_executor.execute_query(
            query, params, fetch_all=True, context="get_conversation_history"
        )

class MedicalReportProcessor:
    """Processes medical reports and test data"""
    
    def __init__(self, query_executor: QueryExecutor):
        self.query_executor = query_executor
    
    def save_medical_test(
        self,
        patient_id: str,
        test_name: str,
        test_value: Union[str, float],
        test_unit: str = "",
        reference_range: str = "",
        is_abnormal: bool = False,
        collection_date: str = None
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Save a medical test result
        
        Args:
            patient_id: Patient identifier
            test_name: Name of the test
            test_value: Test value
            test_unit: Test unit
            reference_range: Reference range
            is_abnormal: Whether the value is abnormal
            collection_date: Date of collection
            
        Returns:
            Tuple of (success, test_id, error_message)
        """
        # Validate the test value
        is_valid, validation_msg = MedicalDataProcessor.validate_medical_value(
            test_name, test_value, test_unit
        )
        
        if not is_valid:
            return False, None, f"Invalid test value: {validation_msg}"
        
        # Normalize test name
        normalized_name = MedicalDataProcessor.normalize_test_name(test_name)
        
        query = """
        INSERT INTO medical_tests (
            patient_id, test_name, test_value, test_unit, 
            reference_range, is_abnormal, collection_date, created_at
        )
        VALUES (
            :patient_id, :test_name, :test_value, :test_unit,
            :reference_range, :is_abnormal, :collection_date, datetime('now')
        )
        """
        
        params = {
            'patient_id': patient_id,
            'test_name': normalized_name,
            'test_value': str(test_value),
            'test_unit': test_unit,
            'reference_range': reference_range,
            'is_abnormal': is_abnormal,
            'collection_date': collection_date
        }
        
        return self.query_executor.execute_insert(
            query, params, f"save_medical_test_{normalized_name}"
        )
    
    def get_patient_tests(
        self,
        patient_id: str,
        test_category: str = None,
        limit: int = 100
    ) -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
        """
        Get medical tests for a patient
        
        Args:
            patient_id: Patient identifier
            test_category: Optional test category filter
            limit: Maximum number of tests to retrieve
            
        Returns:
            Tuple of (success, tests, error_message)
        """
        if test_category:
            query = """
            SELECT test_name, test_value, test_unit, reference_range, 
                   is_abnormal, collection_date, created_at
            FROM medical_tests
            WHERE patient_id = :patient_id
            AND test_name LIKE :category_pattern
            ORDER BY created_at DESC
            LIMIT :limit
            """
            params = {
                'patient_id': patient_id,
                'category_pattern': f'%{test_category}%',
                'limit': limit
            }
        else:
            query = """
            SELECT test_name, test_value, test_unit, reference_range,
                   is_abnormal, collection_date, created_at
            FROM medical_tests
            WHERE patient_id = :patient_id
            ORDER BY created_at DESC
            LIMIT :limit
            """
            params = {
                'patient_id': patient_id,
                'limit': limit
            }
        
        return self.query_executor.execute_query(
            query, params, fetch_all=True, context="get_patient_tests"
        )
