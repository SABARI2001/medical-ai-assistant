"""
Reusable validation and data processing utilities for the Medical AI Assistant
"""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, date
from utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class InputValidator:
    """Validates various types of input data"""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """Validate email address format"""
        if not email:
            return False, "Email is required"
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(pattern, email):
            return True, "Valid email"
        else:
            return False, "Invalid email format"
    
    @staticmethod
    def validate_phone(phone: str) -> Tuple[bool, str]:
        """Validate phone number format"""
        if not phone:
            return False, "Phone number is required"
        
        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', phone)
        
        if len(digits_only) < 10:
            return False, "Phone number too short"
        elif len(digits_only) > 15:
            return False, "Phone number too long"
        else:
            return True, "Valid phone number"
    
    @staticmethod
    def validate_date(date_str: str, date_format: str = "%Y-%m-%d") -> Tuple[bool, str, Optional[date]]:
        """Validate date string format"""
        if not date_str:
            return False, "Date is required", None
        
        try:
            parsed_date = datetime.strptime(date_str, date_format).date()
            return True, "Valid date", parsed_date
        except ValueError:
            return False, f"Invalid date format. Expected: {date_format}", None
    
    @staticmethod
    def validate_uuid(uuid_str: str) -> Tuple[bool, str]:
        """Validate UUID format"""
        if not uuid_str:
            return False, "UUID is required"
        
        try:
            uuid.UUID(uuid_str)
            return True, "Valid UUID"
        except ValueError:
            return False, "Invalid UUID format"
    
    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: List[str]) -> Tuple[bool, str]:
        """Validate file extension"""
        if not filename:
            return False, "Filename is required"
        
        if not allowed_extensions:
            return False, "No allowed extensions specified"
        
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if file_ext in [ext.lower().lstrip('.') for ext in allowed_extensions]:
            return True, "Valid file extension"
        else:
            return False, f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}"

class MedicalDataValidator:
    """Validates medical data and test results"""
    
    # Medical reference ranges
    REFERENCE_RANGES = {
        'glucose': (70, 140),  # mg/dL
        'glucose_fasting': (70, 100),  # mg/dL
        'glucose_random': (70, 140),  # mg/dL
        'hba1c': (4.0, 6.0),  # %
        'cholesterol_total': (0, 200),  # mg/dL
        'cholesterol_hdl': (40, 100),  # mg/dL (male), (50, 100) (female)
        'cholesterol_ldl': (0, 100),  # mg/dL
        'triglycerides': (0, 150),  # mg/dL
        'creatinine': (0.6, 1.2),  # mg/dL (male), (0.5, 1.1) (female)
        'bun': (7, 20),  # mg/dL
        'urea': (15, 45),  # mg/dL
        'uric_acid': (3.5, 7.2),  # mg/dL (male), (2.6, 6.0) (female)
        'tsh': (0.4, 4.0),  # μIU/mL
        't3': (80, 200),  # ng/dL
        't4': (4.5, 12.0),  # μg/dL
        'vitamin_d': (30, 100),  # ng/mL
        'sgot_ast': (10, 40),  # U/L
        'sgpt_alt': (10, 40),  # U/L
        'ggt': (8, 38),  # U/L (male), (5, 27) (female)
        'alkaline_phosphatase': (44, 147),  # U/L
        'bilirubin_total': (0.3, 1.2),  # mg/dL
        'bilirubin_direct': (0.0, 0.3),  # mg/dL
        'bilirubin_indirect': (0.2, 0.9),  # mg/dL
        'phosphorus': (2.5, 4.5),  # mg/dL
    }
    
    @staticmethod
    def validate_medical_value(test_name: str, value: Union[str, float], unit: str = "", 
                             patient_sex: str = None) -> Tuple[bool, str, bool]:
        """
        Validate medical test value against reference ranges
        
        Args:
            test_name: Name of the medical test
            value: Test value
            unit: Test unit
            patient_sex: Patient sex for sex-specific ranges
            
        Returns:
            Tuple of (is_valid, message, is_abnormal)
        """
        try:
            # Convert value to float
            if isinstance(value, str):
                # Extract numeric value from string
                numeric_match = re.search(r'[\d.]+', value)
                if not numeric_match:
                    return False, f"Could not extract numeric value from '{value}'", False
                value = float(numeric_match.group())
            
            # Normalize test name for lookup
            test_key = test_name.lower().replace(' ', '_').replace('-', '_')
            
            # Get reference range
            if test_key in MedicalDataValidator.REFERENCE_RANGES:
                min_val, max_val = MedicalDataValidator.REFERENCE_RANGES[test_key]
                
                # Check for sex-specific ranges
                if test_key == 'cholesterol_hdl' and patient_sex:
                    if patient_sex.lower() == 'female':
                        min_val = 50
                
                # Check if value is within range
                if min_val <= value <= max_val:
                    return True, f"Value {value} is within normal range ({min_val}-{max_val})", False
                else:
                    return True, f"Value {value} is outside normal range ({min_val}-{max_val})", True
            else:
                # No reference range available, just validate format
                if value < 0:
                    return False, "Value cannot be negative", False
                if value > 10000:  # Arbitrary upper limit
                    return False, "Value appears unreasonably high", False
                
                return True, f"Value {value} validated (no reference range available)", False
                
        except (ValueError, TypeError) as e:
            return False, f"Invalid value format: {str(e)}", False
    
    @staticmethod
    def validate_patient_info(patient_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate patient information
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Validate required fields
        required_fields = ['name', 'patient_id']
        for field in required_fields:
            if not patient_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate patient ID format
        if patient_data.get('patient_id'):
            if len(patient_data['patient_id']) < 3:
                errors.append("Patient ID too short")
        
        # Validate age if provided
        if patient_data.get('age'):
            try:
                age = int(patient_data['age'])
                if age < 0 or age > 150:
                    errors.append("Invalid age range")
            except (ValueError, TypeError):
                errors.append("Invalid age format")
        
        # Validate sex if provided
        if patient_data.get('sex'):
            sex = patient_data['sex'].lower()
            if sex not in ['male', 'female', 'm', 'f']:
                errors.append("Invalid sex value")
        
        # Validate dates if provided
        if patient_data.get('collection_date'):
            is_valid, msg, _ = InputValidator.validate_date(patient_data['collection_date'])
            if not is_valid:
                errors.append(f"Invalid collection date: {msg}")
        
        return len(errors) == 0, errors

class ResponseValidator:
    """Validates LLM responses and content"""
    
    @staticmethod
    def validate_response_length(response: str, min_length: int = 10, max_length: int = 10000) -> Tuple[bool, str]:
        """Validate response length"""
        if not response:
            return False, "Response is empty"
        
        length = len(response.strip())
        
        if length < min_length:
            return False, f"Response too short (minimum {min_length} characters)"
        
        if length > max_length:
            return False, f"Response too long (maximum {max_length} characters)"
        
        return True, f"Response length valid ({length} characters)"
    
    @staticmethod
    def validate_medical_response(response: str, required_terms: List[str] = None) -> Tuple[bool, str]:
        """Validate medical response content"""
        if not response:
            return False, "Response is empty"
        
        response_lower = response.lower()
        
        # Check for required medical terms
        if required_terms:
            missing_terms = []
            for term in required_terms:
                if term.lower() not in response_lower:
                    missing_terms.append(term)
            
            if missing_terms:
                return False, f"Missing required terms: {', '.join(missing_terms)}"
        
        # Check for inappropriate content
        inappropriate_patterns = [
            r'\b(not a doctor|not medical advice|consult your doctor)\b',
            r'\b(diagnosis|diagnose|treat|treatment)\b',
            r'\b(emergency|urgent|call 911)\b'
        ]
        
        for pattern in inappropriate_patterns:
            if re.search(pattern, response_lower):
                return False, f"Response contains inappropriate medical content: {pattern}"
        
        return True, "Medical response content valid"
    
    @staticmethod
    def validate_paragraph_structure(response: str, expected_paragraphs: int = None) -> Tuple[bool, str]:
        """Validate paragraph structure for detailed responses"""
        if not response:
            return False, "Response is empty"
        
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        
        if expected_paragraphs and len(paragraphs) != expected_paragraphs:
            return False, f"Expected {expected_paragraphs} paragraphs, found {len(paragraphs)}"
        
        if len(paragraphs) < 1:
            return False, "Response must contain at least one paragraph"
        
        # Check paragraph lengths
        for i, para in enumerate(paragraphs):
            if len(para) < 20:
                return False, f"Paragraph {i+1} is too short (minimum 20 characters)"
        
        return True, f"Paragraph structure valid ({len(paragraphs)} paragraphs)"

class DataSanitizer:
    """Sanitizes and cleans data"""
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 1000) -> str:
        """Sanitize text input"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove potentially harmful characters
        text = re.sub(r'[<>"\']', '', text)
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        if not filename:
            return "unnamed_file"
        
        # Remove or replace unsafe characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Ensure it's not empty
        if not filename:
            filename = "unnamed_file"
        
        return filename
    
    @staticmethod
    def sanitize_sql_input(input_str: str) -> str:
        """Sanitize input for SQL queries (basic protection)"""
        if not input_str:
            return ""
        
        # Remove SQL injection patterns
        dangerous_patterns = [
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
            r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',
            r'(\b(OR|AND)\s+\w+\s*=\s*\w+)',
            r'(\'|\"|;|--|\/\*|\*\/)'
        ]
        
        for pattern in dangerous_patterns:
            input_str = re.sub(pattern, '', input_str, flags=re.IGNORECASE)
        
        return input_str.strip()

class ConfigValidator:
    """Validates application configuration"""
    
    @staticmethod
    def validate_api_keys(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate API key configuration"""
        errors = []
        
        # Check required API keys
        required_keys = ['GROQ_API_KEY', 'SERPER_API_KEY']
        for key in required_keys:
            if not config.get(key):
                errors.append(f"Missing required API key: {key}")
        
        # Validate key formats
        if config.get('GROQ_API_KEY'):
            is_valid, msg = InputValidator.validate_groq_key(config['GROQ_API_KEY'])
            if not is_valid:
                errors.append(f"Invalid Groq API key: {msg}")
        
        if config.get('GOOGLE_API_KEY'):
            is_valid, msg = InputValidator.validate_google_key(config['GOOGLE_API_KEY'])
            if not is_valid:
                errors.append(f"Invalid Google API key: {msg}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_database_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate database configuration"""
        errors = []
        
        if not config.get('DATABASE_URL'):
            errors.append("Missing database URL")
        else:
            db_url = config['DATABASE_URL']
            if not db_url.startswith(('sqlite:///', 'postgresql://', 'mysql://')):
                errors.append("Invalid database URL format")
        
        return len(errors) == 0, errors

# Global validator instances
input_validator = InputValidator()
medical_validator = MedicalDataValidator()
response_validator = ResponseValidator()
data_sanitizer = DataSanitizer()
config_validator = ConfigValidator()
