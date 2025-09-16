"""
Comprehensive error handling utilities for the Medical AI Assistant
"""

import logging
import traceback
import functools
from typing import Any, Callable, Dict, Optional, Union
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

class ErrorHandler:
    """Centralized error handling for the application"""
    
    @staticmethod
    def log_error(error: Exception, context: str = "", additional_info: Dict = None) -> str:
        """
        Log error with context and return error ID for tracking
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            additional_info: Additional information to log
            
        Returns:
            str: Error ID for tracking
        """
        error_id = f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(error)}"
        
        error_details = {
            'error_id': error_id,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc()
        }
        
        if additional_info:
            error_details.update(additional_info)
        
        logger.error(f"Error {error_id}: {error_details}")
        
        return error_id
    
    @staticmethod
    def safe_execute(func: Callable, *args, default_return: Any = None, 
                    context: str = "", **kwargs) -> tuple[Any, Optional[str]]:
        """
        Safely execute a function with error handling
        
        Args:
            func: Function to execute
            *args: Function arguments
            default_return: Value to return if function fails
            context: Context for error logging
            **kwargs: Function keyword arguments
            
        Returns:
            tuple: (result, error_id) where error_id is None if successful
        """
        try:
            result = func(*args, **kwargs)
            return result, None
        except Exception as e:
            error_id = ErrorHandler.log_error(e, context, {
                'function': func.__name__,
                'args': str(args)[:200],  # Truncate long args
                'kwargs': str(kwargs)[:200]
            })
            return default_return, error_id

def handle_errors(context: str = "", default_return: Any = None, 
                 reraise: bool = False, log_level: str = "ERROR"):
    """
    Decorator for automatic error handling
    
    Args:
        context: Context for error logging
        default_return: Value to return if function fails
        reraise: Whether to reraise the exception after logging
        log_level: Logging level for errors
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_id = ErrorHandler.log_error(e, context, {
                    'function': func.__name__,
                    'args': str(args)[:200],
                    'kwargs': str(kwargs)[:200]
                })
                
                if log_level.upper() == "WARNING":
                    logger.warning(f"Warning in {func.__name__}: {str(e)} (Error ID: {error_id})")
                else:
                    logger.error(f"Error in {func.__name__}: {str(e)} (Error ID: {error_id})")
                
                if reraise:
                    raise
                
                return default_return
        return wrapper
    return decorator

def validate_api_response(response: Any, expected_keys: list = None) -> tuple[bool, Optional[str]]:
    """
    Validate API response structure
    
    Args:
        response: API response to validate
        expected_keys: List of expected keys in response
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        if response is None:
            return False, "Response is None"
        
        if isinstance(response, dict):
            if expected_keys:
                missing_keys = [key for key in expected_keys if key not in response]
                if missing_keys:
                    return False, f"Missing keys: {missing_keys}"
        
        return True, None
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                    backoff_factor: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Decorator for retrying failed operations
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. Retrying in {current_delay}s...")
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        raise last_exception
            
            raise last_exception
        return wrapper
    return decorator

class DatabaseError(Exception):
    """Custom exception for database-related errors"""
    pass

class APIError(Exception):
    """Custom exception for API-related errors"""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def create_error_response(error_type: str, message: str, error_id: str = None) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        error_type: Type of error
        message: Error message
        error_id: Optional error ID for tracking
        
    Returns:
        dict: Standardized error response
    """
    return {
        'success': False,
        'error_type': error_type,
        'message': message,
        'error_id': error_id,
        'timestamp': datetime.now().isoformat()
    }

def create_success_response(data: Any = None, message: str = "Operation successful") -> Dict[str, Any]:
    """
    Create standardized success response
    
    Args:
        data: Response data
        message: Success message
        
    Returns:
        dict: Standardized success response
    """
    return {
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
