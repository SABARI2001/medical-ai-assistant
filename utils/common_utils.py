"""
Common utility functions for the Medical AI Assistant
"""

import os
import re
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

class TextProcessor:
    """Utility class for text processing operations"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep medical symbols
        text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\[\]\{\}\+\-\*\/\=\<\>\&\%\$\#\@\!]', '', text)
        
        return text
    
    @staticmethod
    def extract_keywords(text: str, min_length: int = 3) -> List[str]:
        """Extract keywords from text"""
        if not text:
            return []
        
        # Convert to lowercase and split
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter by length and common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}
        
        keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
        
        return list(set(keywords))  # Remove duplicates
    
    @staticmethod
    def extract_medical_terms(text: str) -> List[str]:
        """Extract medical terms from text"""
        if not text:
            return []
        
        # Common medical term patterns
        medical_patterns = [
            r'\b[A-Z][a-z]+\s+(?:syndrome|disease|disorder|condition|deficiency|excess)\b',
            r'\b(?:glucose|cholesterol|triglycerides|HDL|LDL|VLDL|BUN|creatinine|uric acid|bilirubin|SGOT|SGPT|GGT|TSH|T3|T4|vitamin D)\b',
            r'\b(?:mg/dL|U/L|ng/mL|μIU/mL|μg/dL)\b',
            r'\b(?:normal|abnormal|elevated|decreased|high|low|borderline)\b'
        ]
        
        medical_terms = []
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            medical_terms.extend(matches)
        
        return list(set(medical_terms))

class FileProcessor:
    """Utility class for file processing operations"""
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Generate hash for file to detect duplicates"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating file hash: {str(e)}")
            return ""
    
    @staticmethod
    def validate_file_type(file_path: str, allowed_extensions: List[str]) -> bool:
        """Validate file extension"""
        if not file_path:
            return False
        
        _, ext = os.path.splitext(file_path.lower())
        return ext in allowed_extensions
    
    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """Get file size in MB"""
        try:
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 * 1024)
        except Exception as e:
            logger.error(f"Error getting file size: {str(e)}")
            return 0.0

class DataValidator:
    """Utility class for data validation"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number format"""
        pattern = r'^[\+]?[1-9][\d]{0,15}$'
        return bool(re.match(pattern, phone.replace(' ', '').replace('-', '')))
    
    @staticmethod
    def validate_medical_value(value: Union[str, float, int], 
                             min_val: float = None, max_val: float = None) -> bool:
        """Validate medical test values"""
        try:
            if isinstance(value, str):
                # Extract numeric value from string
                numeric_value = re.findall(r'[\d.]+', value)
                if not numeric_value:
                    return False
                value = float(numeric_value[0])
            
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
            
            return True
        except (ValueError, TypeError):
            return False

class CacheManager:
    """Simple cache manager for frequently accessed data"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Any:
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if datetime.now() > entry['expires_at']:
            del self.cache[key]
            return None
        
        return entry['value']
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['created_at'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'value': value,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=self.ttl_seconds)
        }
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()

class ResponseFormatter:
    """Utility class for formatting responses"""
    
    @staticmethod
    def format_medical_data(data: Dict[str, Any]) -> str:
        """Format medical data for display"""
        if not data:
            return "No medical data available"
        
        formatted = []
        
        # Patient information
        if 'patient_name' in data:
            formatted.append(f"**Patient:** {data['patient_name']}")
        if 'patient_age' in data and 'patient_sex' in data:
            formatted.append(f"**Age/Sex:** {data['patient_age']}/{data['patient_sex']}")
        if 'collection_date' in data:
            formatted.append(f"**Collection Date:** {data['collection_date']}")
        
        # Test results
        if 'tests' in data and data['tests']:
            formatted.append("\n**Test Results:**")
            for test in data['tests']:
                test_line = f"- {test.get('test_name', 'Unknown')}: {test.get('test_value', 'N/A')}"
                if test.get('test_unit'):
                    test_line += f" {test['test_unit']}"
                if test.get('reference_range'):
                    test_line += f" (Ref: {test['reference_range']})"
                if test.get('is_abnormal'):
                    test_line += " **[ABNORMAL]**"
                formatted.append(test_line)
        
        return "\n".join(formatted)
    
    @staticmethod
    def format_error_message(error: str, context: str = "") -> str:
        """Format error message for user display"""
        if context:
            return f"Error in {context}: {error}"
        return f"Error: {error}"
    
    @staticmethod
    def format_success_message(message: str, data: Any = None) -> str:
        """Format success message"""
        if data:
            return f"{message}\n\nData: {json.dumps(data, indent=2, default=str)}"
        return message

class ConfigHelper:
    """Helper class for configuration management"""
    
    @staticmethod
    def get_environment_variable(key: str, default: str = "", required: bool = False) -> str:
        """Get environment variable with validation"""
        value = os.getenv(key, default)
        
        if required and not value:
            raise ValueError(f"Required environment variable {key} is not set")
        
        return value
    
    @staticmethod
    def validate_required_config(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
        """Validate that required configuration keys are present"""
        missing_keys = []
        for key in required_keys:
            if key not in config or not config[key]:
                missing_keys.append(key)
        return missing_keys

# Global cache instance
global_cache = CacheManager()

# Common validation patterns
MEDICAL_VALUE_PATTERNS = {
    'glucose': (0, 1000),  # mg/dL
    'cholesterol': (0, 1000),  # mg/dL
    'triglycerides': (0, 2000),  # mg/dL
    'creatinine': (0, 20),  # mg/dL
    'bun': (0, 100),  # mg/dL
    'tsh': (0, 100),  # μIU/mL
    't3': (0, 1000),  # ng/dL
    't4': (0, 50),  # μg/dL
    'vitamin_d': (0, 200)  # ng/mL
}

def get_medical_reference_range(test_name: str) -> Tuple[float, float]:
    """Get reference range for medical test"""
    test_lower = test_name.lower().replace(' ', '_')
    return MEDICAL_VALUE_PATTERNS.get(test_lower, (0, 1000))

def is_abnormal_value(test_name: str, value: float) -> bool:
    """Check if medical test value is abnormal"""
    try:
        min_val, max_val = get_medical_reference_range(test_name)
        return value < min_val or value > max_val
    except (ValueError, TypeError):
        return False
