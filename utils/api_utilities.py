"""
Reusable API interaction utilities for the Medical AI Assistant
"""

import logging
import time
import requests
from typing import Any, Dict, List, Optional, Tuple, Union
from utils.error_handler import ErrorHandler, retry_on_failure

logger = logging.getLogger(__name__)

class APIResponseHandler:
    """Handles API responses with standardized error handling and retry logic"""
    
    @staticmethod
    @retry_on_failure(max_retries=3, delay=1.0, backoff_factor=2.0)
    def make_request(
        method: str,
        url: str,
        headers: Dict[str, str] = None,
        data: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
        timeout: int = 30,
        context: str = "api_request"
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Make an HTTP request with error handling and retry logic
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            headers: Request headers
            data: Request data
            params: Query parameters
            timeout: Request timeout in seconds
            context: Context for error logging
            
        Returns:
            Tuple of (success, response_data, error_message)
        """
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers or {},
                json=data,
                params=params,
                timeout=timeout
            )
            
            # Check if request was successful
            if response.status_code == 200:
                try:
                    return True, response.json(), None
                except ValueError:
                    return True, {"content": response.text}, None
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                return False, None, error_msg
                
        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after {timeout} seconds"
            return False, None, error_msg
        except requests.exceptions.ConnectionError:
            error_msg = "Connection error - check network connectivity"
            return False, None, error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            return False, None, error_msg
        except Exception as e:
            error_id = ErrorHandler.log_error(e, context, {
                'method': method,
                'url': url,
                'timeout': timeout
            })
            return False, None, f"Unexpected error (ID: {error_id}): {str(e)}"

class LLMResponseProcessor:
    """Processes and validates LLM responses"""
    
    @staticmethod
    def validate_response(response: str, min_length: int = 10) -> Tuple[bool, str]:
        """
        Validate LLM response quality
        
        Args:
            response: LLM response text
            min_length: Minimum response length
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        if not response or not isinstance(response, str):
            return False, "Empty or invalid response"
        
        if len(response.strip()) < min_length:
            return False, f"Response too short (minimum {min_length} characters)"
        
        # Check for common error patterns
        error_patterns = [
            "i'm sorry, i cannot",
            "i cannot provide",
            "i'm not able to",
            "i don't have access",
            "error occurred",
            "something went wrong"
        ]
        
        response_lower = response.lower()
        for pattern in error_patterns:
            if pattern in response_lower:
                return False, f"Response contains error pattern: {pattern}"
        
        return True, "Response is valid"
    
    @staticmethod
    def clean_response(response: str) -> str:
        """
        Clean and normalize LLM response
        
        Args:
            response: Raw LLM response
            
        Returns:
            Cleaned response text
        """
        if not response:
            return ""
        
        # Remove common prefixes/suffixes
        response = response.strip()
        
        # Remove markdown formatting if not needed
        response = response.replace("**", "").replace("*", "")
        
        # Ensure proper sentence endings
        if response and not response.endswith(('.', '!', '?')):
            response += "."
        
        return response

class RateLimitHandler:
    """Handles API rate limiting with exponential backoff"""
    
    def __init__(self, max_retries: int = 5, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_count = 0
    
    def should_retry(self, error_message: str) -> bool:
        """Check if error indicates rate limiting"""
        rate_limit_indicators = [
            "rate limit",
            "too many requests",
            "429",
            "quota exceeded",
            "throttled"
        ]
        
        error_lower = error_message.lower()
        return any(indicator in error_lower for indicator in rate_limit_indicators)
    
    def get_retry_delay(self) -> float:
        """Calculate delay for next retry attempt"""
        if self.retry_count >= self.max_retries:
            return 0
        
        delay = self.base_delay * (2 ** self.retry_count)
        self.retry_count += 1
        return delay
    
    def reset(self):
        """Reset retry counter"""
        self.retry_count = 0

class APIKeyValidator:
    """Validates API keys for different services"""
    
    @staticmethod
    def validate_groq_key(api_key: str) -> Tuple[bool, str]:
        """Validate Groq API key format"""
        if not api_key:
            return False, "API key is empty"
        
        if not api_key.startswith("gsk_"):
            return False, "Invalid Groq API key format (should start with 'gsk_')"
        
        if len(api_key) < 50:
            return False, "API key appears to be too short"
        
        return True, "Valid Groq API key"
    
    @staticmethod
    def validate_openai_key(api_key: str) -> Tuple[bool, str]:
        """Validate OpenAI API key format"""
        if not api_key:
            return False, "API key is empty"
        
        if not api_key.startswith("sk-"):
            return False, "Invalid OpenAI API key format (should start with 'sk-')"
        
        if len(api_key) < 40:
            return False, "API key appears to be too short"
        
        return True, "Valid OpenAI API key"
    
    @staticmethod
    def validate_google_key(api_key: str) -> Tuple[bool, str]:
        """Validate Google API key format"""
        if not api_key:
            return False, "API key is empty"
        
        if not api_key.startswith("AIza"):
            return False, "Invalid Google API key format (should start with 'AIza')"
        
        if len(api_key) < 30:
            return False, "API key appears to be too short"
        
        return True, "Valid Google API key"
    
    @staticmethod
    def validate_serper_key(api_key: str) -> Tuple[bool, str]:
        """Validate Serper API key format"""
        if not api_key:
            return False, "API key is empty"
        
        if len(api_key) < 20:
            return False, "API key appears to be too short"
        
        return True, "Valid Serper API key"

class ResponseFormatter:
    """Formats responses for different output modes"""
    
    @staticmethod
    def format_concise_response(response: str) -> str:
        """Format response for concise mode"""
        if not response:
            return "I don't have enough information to provide a response."
        
        # Ensure single paragraph
        response = response.replace('\n\n', ' ').replace('\n', ' ')
        response = ' '.join(response.split())  # Remove extra spaces
        
        return response.strip()
    
    @staticmethod
    def format_detailed_response(response: str) -> str:
        """Format response for detailed mode"""
        if not response:
            return "I don't have enough information to provide a detailed response."
        
        # Ensure proper paragraph structure
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 3:
            # If less than 3 paragraphs, create them
            content = ' '.join(paragraphs)
            # Split into roughly equal parts
            words = content.split()
            words_per_para = len(words) // 3
            
            paragraphs = []
            for i in range(3):
                start = i * words_per_para
                end = start + words_per_para if i < 2 else len(words)
                para = ' '.join(words[start:end])
                if para.strip():
                    paragraphs.append(para.strip())
        
        return '\n\n'.join(paragraphs)
    
    @staticmethod
    def format_technical_response(response: str, web_context: List[str] = None) -> str:
        """Format response for technical mode with web context"""
        if not response:
            return "I don't have enough information to provide a technical response."
        
        # Add web context if available
        if web_context:
            web_info = "\n\n**Additional Research:**\n" + "\n".join(f"- {info}" for info in web_context[:3])
            response += web_info
        
        return response

# Global instances for reuse
api_handler = APIResponseHandler()
response_processor = LLMResponseProcessor()
rate_limiter = RateLimitHandler()
key_validator = APIKeyValidator()
formatter = ResponseFormatter()
