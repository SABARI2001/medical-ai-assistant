import json
import re
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from models.llm import LLMManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalReportExtractor:
    """Extract structured medical test data from uploaded documents using comprehensive regex patterns"""

    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        
        # Define all medical tests to extract
        self.medical_tests = {
            "Patient": "Patient",
            "Age": "Age",
            "Sex": "Sex",
            "Patient ID": "Patient ID",
            "Collection Date": "Collection Date",
            "Report Date": "Report Date",

            # Biochemistry
            "glucose_rbs": "Glucose - Random (RBS)",
            "blood_urea_nitrogen": "Blood Urea Nitrogen (BUN)",
            "urea": "Urea",
            "creatinine": "Creatinine",
            "uric_acid": "Uric Acid",
            "phosphorus": "Phosphorus",
            
            # Liver Function Tests
            "bilirubin_total": "Bilirubin (Total)",
            "bilirubin_direct": "Bilirubin (Direct)",
            "bilirubin_indirect": "Bilirubin (Indirect)",
            "sgot_ast": "SGOT/AST (Aspartate Aminotransferase)",
            "sgpt_alt": "SGPT/ALT (Alanine Aminotransferase)",
            "ggt": "GGT (Gamma Glutamyl Transpeptidase)",
            "alkaline_phosphatase": "Alkaline Phosphatase (SAP)",
            "total_protein": "Total Protein",
            "albumin": "Albumin",
            "globulin": "Globulin",
            "ag_ratio": "A/G Ratio",
            
            # Lipid Profile
            "cholesterol_total": "Cholesterol (Total)",
            "triglycerides": "Triglycerides",
            "hdl_cholesterol": "HDL Cholesterol",
            "ldl_cholesterol": "LDL Cholesterol",
            "vldl_cholesterol": "VLDL Cholesterol",
            "non_hdl_cholesterol": "Non-HDL Cholesterol",
            "total_cholesterol_hdl_ratio": "Total Cholesterol/HDL Ratio",
            "triglyceride_hdl_ratio": "Triglyceride/HDL Cholesterol Ratio",
            "ldl_hdl_ratio": "LDL/HDL Cholesterol Ratio",
            
            # Immunoassay
            "vitamin_d_25_hydroxy": "Vitamin D (25-Hydroxy Vit D)",
            
            # Thyroid Profile
            "t3_total": "T3 (Triiodothyronine) - Total",
            "t4_total": "T4 (Thyroxine) - Total",
            "tsh": "TSH (Thyroid Stimulating Hormone)"
        }

    def parse_date(self, date_string: str) -> Optional[str]:
        """Parse date string to standard format"""
        if not date_string:
            return None
            
        # Common date patterns
        patterns = [
            (r'(\d{2}/\d{2}/\d{4})', '%d/%m/%Y'),
            (r'(\d{1,2}/\d{1,2}/\d{4})', '%d/%m/%Y'),
            (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),
        ]
        
        for pattern, date_format in patterns:
            match = re.search(pattern, date_string)
            if match:
                try:
                    date_part = match.group(1)
                    parsed_date = datetime.strptime(date_part, date_format)
                    return parsed_date.strftime('%d/%m/%Y')
                except ValueError:
                    continue
        
        logger.warning(f"Could not parse date: {date_string}")
        return date_string

    def extract_numerical_value(self, text: str, patterns: list) -> Optional[float]:
        """Extract numerical value using multiple regex patterns"""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    # Get the last group which should be the numerical value
                    value_str = match.groups()[-1]
                    return float(value_str)
                except (ValueError, IndexError):
                    continue
        return None

    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract all medical test values and patient info from text"""
        logger.info("Starting comprehensive medical data extraction...")
        
        # Initialize result structure
        result = {
            "patient": {
                "name": "",
                "age": "",
                "sex": "",
                "patientId": "",
                "collectionDate": "",
                "reportDate": ""
            },
            "biochemistry": [],
            "liverFunction": [],
            "lipidProfile": [],
            "thyroidProfile": [],
            "immunoassay": [],
            "other": []
        }

        # Extract patient information with enhanced patterns
        # Patient Name
        name_patterns = [
            r'Name\s*:?\s*([A-Za-z\s\.]+?)(?:\n|PID|Age|Patient)',
            r'Patient\s*Name\s*:?\s*([A-Za-z\s\.]+?)(?:\n|PID|Age)',
            r'Mr[s]?\.\s*([A-Za-z\s\.]+?)(?:\n|PID|Age)',
            r'Name\s*:?\s*([A-Za-z\s\.]+?)(?=\s*\n|\s*PID|\s*Age)',
            r'Patient\s*:\s*([A-Za-z\s\.]+?)(?:\n|PID|Age)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                name = match.group(1).strip()
                # Clean up common prefixes
                name = re.sub(r'^(Mr\.|Mrs\.|Ms\.|Dr\.)\s*', '', name, flags=re.IGNORECASE)
                result["patient"]["name"] = name
                logger.info(f"Extracted patient name: {name}")
                break

        # Patient ID with more patterns
        pid_patterns = [
            r'PID\s*No\.?\s*:?\s*([A-Za-z0-9]+)',
            r'Patient\s*ID\s*:?\s*([A-Za-z0-9]+)',
            r'PID\s*:?\s*([A-Za-z0-9]+)',
            r'ID\s*No\.?\s*:?\s*([A-Za-z0-9]+)',
            r'Registration\s*No\.?\s*:?\s*([A-Za-z0-9]+)',
            r'Reg\.?\s*No\.?\s*:?\s*([A-Za-z0-9]+)'
        ]
        
        for pattern in pid_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["patient"]["patientId"] = match.group(1).strip()
                logger.info(f"Extracted patient ID: {result['patient']['patientId']}")
                break

        # Age and Sex with enhanced patterns
        age_sex_patterns = [
            r'Age[/\s]*Sex\s*:?\s*(\d+)\s*Year.*?(Male|Female)',
            r'Age\s*:?\s*(\d+).*?Sex\s*:?\s*(Male|Female)',
            r'(\d+)\s*Year.*?(Male|Female)',
            r'Age\s*:\s*(\d+).*?Gender\s*:\s*(Male|Female)',
            r'(\d+)\s*[Yy]ear.*?[Ss]ex.*?(Male|Female)',
            r'Age\s*/\s*Sex\s*:?\s*(\d+).*?(Male|Female)',
            r'(\d+)\s*[Yy]rs?.*?(Male|Female)',
            r'(\d+)\s*[Yy]ear\(s\).*?(Male|Female)'
        ]
        
        for pattern in age_sex_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["patient"]["age"] = match.group(1)
                result["patient"]["sex"] = match.group(2)
                logger.info(f"Extracted age/sex: {result['patient']['age']}/{result['patient']['sex']}")
                break

        # Collection Date with multiple patterns
        collection_patterns = [
            r'Collection\s*(?:Date|On)\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})',
            r'Collected\s*(?:on|at)\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})',
            r'Sample\s*(?:Date|Collection)\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})',
            r'Date\s*of\s*Collection\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})'
        ]
        
        for pattern in collection_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["patient"]["collectionDate"] = self.parse_date(match.group(1))
                logger.info(f"Extracted collection date: {result['patient']['collectionDate']}")
                break

        # Report Date with multiple patterns
        report_patterns = [
            r'Report\s*(?:Date|On)\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})',
            r'Reported\s*(?:on|at)\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})',
            r'Date\s*of\s*Report\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})',
            r'Report\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})'
        ]
        
        for pattern in report_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["patient"]["reportDate"] = self.parse_date(match.group(1))
                logger.info(f"Extracted report date: {result['patient']['reportDate']}")
                break

        # Define comprehensive regex patterns for each medical test
        test_patterns = {
            # Biochemistry
            "glucose_rbs": [
                r'Glucose.*?(?:Random|RBS).*?(\d+\.?\d*)\s*mg/dL',
                r'Glucose.*?Fasting.*?(\d+\.?\d*)\s*mg/dL',
                r'Glucose.*?(\d+\.?\d*)\s*mg/dL'
            ],
            "blood_urea_nitrogen": [
                r'Blood\s+Urea\s+Nitrogen.*?(\d+\.?\d*)\s*mg/dL',
                r'BUN.*?(\d+\.?\d*)\s*mg/dL'
            ],
            "urea": [r'Urea[^:]*?(\d+\.?\d*)\s*mg/dL'],
            "creatinine": [r'Creatinine[^:]*?(\d+\.?\d*)\s*mg/dL'],
            "uric_acid": [r'Uric\s+Acid[^:]*?(\d+\.?\d*)\s*mg/dL'],
            "phosphorus": [r'Phosphorus[^:]*?(\d+\.?\d*)\s*mg/dL'],

            # Liver Function Tests
            "bilirubin_total": [
                r'Bilirubin.*?Total[^:]*?(\d+\.?\d*)\s*mg/dL',
                r'Total\s+Bilirubin[^:]*?(\d+\.?\d*)\s*mg/dL'
            ],
            "bilirubin_direct": [
                r'Bilirubin.*?Direct[^:]*?(\d+\.?\d*)\s*mg/dL',
                r'Direct\s+Bilirubin[^:]*?(\d+\.?\d*)\s*mg/dL'
            ],
            "bilirubin_indirect": [
                r'Bilirubin.*?Indirect[^:]*?(\d+\.?\d*)\s*mg/dL',
                r'Indirect\s+Bilirubin[^:]*?(\d+\.?\d*)\s*mg/dL'
            ],
            "sgot_ast": [
                r'SGOT/AST[^:]*?(\d+\.?\d*)\s*U/L',
                r'SGOT[^:]*?(\d+\.?\d*)\s*U/L',
                r'AST[^:]*?(\d+\.?\d*)\s*U/L',
                r'Aspartate\s+Aminotransferase[^:]*?(\d+\.?\d*)\s*U/L'
            ],
            "sgpt_alt": [
                r'SGPT/ALT[^:]*?(\d+\.?\d*)\s*U/L',
                r'SGPT[^:]*?(\d+\.?\d*)\s*U/L',
                r'ALT[^:]*?(\d+\.?\d*)\s*U/L',
                r'Alanine\s+Aminotransferase[^:]*?(\d+\.?\d*)\s*U/L'
            ],
            "ggt": [
                r'GGT[^:]*?(\d+\.?\d*)\s*U/L',
                r'Gamma\s+Glutamyl[^:]*?(\d+\.?\d*)\s*U/L'
            ],
            "alkaline_phosphatase": [
                r'Alkaline\s+Phosphatase[^:]*?(\d+\.?\d*)\s*U/L',
                r'ALP[^:]*?(\d+\.?\d*)\s*U/L',
                r'SAP[^:]*?(\d+\.?\d*)\s*U/L'
            ],
            "total_protein": [r'Total\s+Protein[^:]*?(\d+\.?\d*)\s*g/dL'],
            "albumin": [r'Albumin[^:]*?(\d+\.?\d*)\s*g/dL'],
            "globulin": [r'Globulin[^:]*?(\d+\.?\d*)\s*g/dL'],
            "ag_ratio": [r'A/G\s+Ratio[^:]*?(\d+\.?\d*)'],

            # Lipid Profile
            "cholesterol_total": [
                r'Total\s+Cholesterol[^:]*?(\d+\.?\d*)\s*mg/dL',
                r'Cholesterol.*?Total[^:]*?(\d+\.?\d*)\s*mg/dL',
                r'Cholesterol[^:]*?(\d+\.?\d*)\s*mg/dL'
            ],
            "triglycerides": [r'Triglycerides[^:]*?(\d+\.?\d*)\s*mg/dL'],
            "hdl_cholesterol": [
                r'HDL\s+Cholesterol[^:]*?(\d+\.?\d*)\s*mg/dL',
                r'HDL[^:]*?(\d+\.?\d*)\s*mg/dL'
            ],
            "ldl_cholesterol": [
                r'LDL\s+Cholesterol[^:]*?(\d+\.?\d*)\s*mg/dL',
                r'LDL[^:]*?(\d+\.?\d*)\s*mg/dL'
            ],
            "vldl_cholesterol": [
                r'VLDL\s+Cholesterol[^:]*?(\d+\.?\d*)\s*mg/dL',
                r'VLDL[^:]*?(\d+\.?\d*)\s*mg/dL'
            ],
            "non_hdl_cholesterol": [
                r'Non-HDL\s+Cholesterol[^:]*?(\d+\.?\d*)\s*mg/dL',
                r'Non\s+HDL[^:]*?(\d+\.?\d*)\s*mg/dL'
            ],
            "total_cholesterol_hdl_ratio": [
                r'Total\s+Cholesterol/HDL\s+Ratio[^:]*?(\d+\.?\d*)',
                r'TC/HDL[^:]*?(\d+\.?\d*)'
            ],
            "triglyceride_hdl_ratio": [
                r'Triglyceride/HDL.*?Ratio[^:]*?(\d+\.?\d*)',
                r'TG/HDL[^:]*?(\d+\.?\d*)'
            ],
            "ldl_hdl_ratio": [
                r'LDL/HDL.*?Ratio[^:]*?(\d+\.?\d*)',
                r'LDL\s+HDL\s+Ratio[^:]*?(\d+\.?\d*)'
            ],

            # Immunoassay
            "vitamin_d_25_hydroxy": [
                r'Vitamin\s+D.*?25.*?Hydroxy[^:]*?(\d+\.?\d*)\s*ng/mL',
                r'25.*?Hydroxy.*?Vitamin.*?D[^:]*?(\d+\.?\d*)\s*ng/mL',
                r'Vitamin\s+D[^:]*?(\d+\.?\d*)\s*ng/mL'
            ],

            # Thyroid Profile
            "t3_total": [
                r'T3.*?Total[^:]*?(\d+\.?\d*)\s*ng/mL',
                r'Triiodothyronine.*?Total[^:]*?(\d+\.?\d*)\s*ng/mL',
                r'T3[^:]*?(\d+\.?\d*)\s*ng/mL'
            ],
            "t4_total": [
                r'T4.*?Total[^:]*?(\d+\.?\d*)\s*µg/dL',
                r'Thyroxine.*?Total[^:]*?(\d+\.?\d*)\s*µg/dL',
                r'T4[^:]*?(\d+\.?\d*)\s*µg/dL'
            ],
            "tsh": [
                r'TSH[^:]*?(\d+\.?\d*)\s*µIU/mL',
                r'Thyroid\s+Stimulating\s+Hormone[^:]*?(\d+\.?\d*)\s*µIU/mL'
            ]
        }

        # Extract all test values and categorize them
        extracted_tests = []
        for test_key, patterns in test_patterns.items():
            value = self.extract_numerical_value(text, patterns)
            if value is not None:
                test_name = self.medical_tests[test_key]
                
                # Determine the category
                category = self._get_test_category(test_key)
                
                test_entry = {
                    "test": test_name,
                    "value": str(value),
                    "unit": self._get_test_unit(test_key),
                    "referenceRange": ""
                }
                
                result[category].append(test_entry)
                extracted_tests.append(f"{test_name}: {value}")
                logger.info(f"Extracted {test_name}: {value}")

        # Try LLM extraction as fallback if we didn't get enough data
        if len(extracted_tests) < 5:
            logger.info("Low extraction count, trying LLM fallback...")
            try:
                llm_result = self._extract_with_llm(text)
                if llm_result and llm_result.get('patient', {}).get('name'):
                    # Merge LLM results with regex results
                    if not result["patient"]["name"]:
                        result["patient"] = llm_result.get("patient", result["patient"])
                    
                    # Add any additional tests found by LLM
                    for category in ['biochemistry', 'liverFunction', 'lipidProfile', 'thyroidProfile', 'immunoassay', 'other']:
                        llm_tests = llm_result.get(category, [])
                        for test in llm_tests:
                            if test.get('test') and test.get('value'):
                                # Check if we already have this test
                                existing = any(t.get('test', '').lower() == test.get('test', '').lower() 
                                             for t in result[category])
                                if not existing:
                                    result[category].append(test)
                                    extracted_tests.append(f"{test.get('test')}: {test.get('value')}")
            except Exception as e:
                logger.warning(f"LLM fallback failed: {str(e)}")

        logger.info(f"Total tests extracted: {len(extracted_tests)}")
        
        # Extract reference ranges for extracted tests
        self._extract_reference_ranges(text, result)
        
        return result

    def _get_test_category(self, test_key: str) -> str:
        """Determine test category from test key"""
        if test_key in ['glucose_rbs', 'blood_urea_nitrogen', 'urea', 'creatinine', 'uric_acid', 'phosphorus']:
            return 'biochemistry'
        elif test_key in ['bilirubin_total', 'bilirubin_direct', 'bilirubin_indirect', 'sgot_ast', 'sgpt_alt', 
                         'ggt', 'alkaline_phosphatase', 'total_protein', 'albumin', 'globulin', 'ag_ratio']:
            return 'liverFunction'
        elif test_key in ['cholesterol_total', 'triglycerides', 'hdl_cholesterol', 'ldl_cholesterol', 
                         'vldl_cholesterol', 'non_hdl_cholesterol', 'total_cholesterol_hdl_ratio', 
                         'triglyceride_hdl_ratio', 'ldl_hdl_ratio']:
            return 'lipidProfile'
        elif test_key in ['t3_total', 't4_total', 'tsh']:
            return 'thyroidProfile'
        elif test_key in ['vitamin_d_25_hydroxy']:
            return 'immunoassay'
        else:
            return 'other'

    def _get_test_unit(self, test_key: str) -> str:
        """Get the unit for a test"""
        unit_map = {
            'glucose_rbs': 'mg/dL',
            'blood_urea_nitrogen': 'mg/dL',
            'urea': 'mg/dL',
            'creatinine': 'mg/dL',
            'uric_acid': 'mg/dL',
            'phosphorus': 'mg/dL',
            'bilirubin_total': 'mg/dL',
            'bilirubin_direct': 'mg/dL',
            'bilirubin_indirect': 'mg/dL',
            'sgot_ast': 'U/L',
            'sgpt_alt': 'U/L',
            'ggt': 'U/L',
            'alkaline_phosphatase': 'U/L',
            'total_protein': 'g/dL',
            'albumin': 'g/dL',
            'globulin': 'g/dL',
            'ag_ratio': '',
            'cholesterol_total': 'mg/dL',
            'triglycerides': 'mg/dL',
            'hdl_cholesterol': 'mg/dL',
            'ldl_cholesterol': 'mg/dL',
            'vldl_cholesterol': 'mg/dL',
            'non_hdl_cholesterol': 'mg/dL',
            'total_cholesterol_hdl_ratio': '',
            'triglyceride_hdl_ratio': '',
            'ldl_hdl_ratio': '',
            'vitamin_d_25_hydroxy': 'ng/mL',
            't3_total': 'ng/mL',
            't4_total': 'µg/dL',
            'tsh': 'µIU/mL'
        }
        return unit_map.get(test_key, '')

    def _extract_reference_ranges(self, text: str, result: Dict[str, Any]):
        """Extract reference ranges for tests"""
        try:
            # Look for patterns like "value mg/dL (reference range)"
            range_pattern = r'(\d+\.?\d*)\s*([a-zA-Z/µ]+)\s*\(([^)]+)\)'
            matches = re.finditer(range_pattern, text, re.IGNORECASE)

            for match in matches:
                value = match.group(1)
                unit = match.group(2)
                ref_range = match.group(3)

                # Find matching test in results
                for category in ['biochemistry', 'liverFunction', 'lipidProfile', 'thyroidProfile', 'immunoassay', 'other']:
                    for test in result[category]:
                        if (test['value'] == value and 
                            test['unit'].lower().replace('µ', 'u') == unit.lower().replace('µ', 'u') and
                            not test['referenceRange']):
                            test['referenceRange'] = ref_range
                            break
        except Exception as e:
            logger.warning(f"Error extracting reference ranges: {str(e)}")

    def _extract_with_llm(self, text: str) -> Optional[Dict[str, Any]]:
        """Fallback LLM extraction method"""
        try:
            prompt = """Extract medical test data from this report. Return JSON with patient info and test results.
            
            Format:
            {
              "patient": {"name": "", "age": "", "sex": "", "patientId": "", "collectionDate": "", "reportDate": ""},
              "biochemistry": [{"test": "", "value": "", "unit": "", "referenceRange": ""}],
              "liverFunction": [...],
              "lipidProfile": [...],
              "thyroidProfile": [...],
              "immunoassay": [...],
              "other": [...]
            }
            
            Text: """ + text

            response = self.llm_manager.generate_response(
                prompt=prompt,
                context=[],
                response_mode="Technical",
                provider="Groq",
                model="groq/compound-mini"
            )

            # Clean and parse JSON
            json_str = self._clean_json_response(response)
            return json.loads(json_str)

        except Exception as e:
            logger.error(f"LLM extraction failed: {str(e)}")
            return None

    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response to extract valid JSON"""
        try:
            # Remove markdown formatting
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*$', '', response)
            
            # Find JSON object
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = response[start:end]
                
                # Remove trailing commas
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                
                return json_str
            
            return "{}"
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {str(e)}")
            return "{}"
