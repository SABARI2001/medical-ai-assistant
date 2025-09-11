"""
Simplified RAG system that works with the improved medical database structure
"""

import logging
from typing import List, Dict, Any
from models.database import DatabaseManager

# Setup logging
logger = logging.getLogger(__name__)

class SimpleMedicalRAG:
    """Simplified RAG system for medical data using wide table format"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def search_medical_data(self, query: str) -> Dict[str, Any]:
        """Search medical data based on query keywords"""
        try:
            contexts = []
            sources = []
            
            query_lower = query.lower()
            
            # Search for patient names
            patient_keywords = self._extract_patient_keywords(query_lower)
            if patient_keywords:
                for keyword in patient_keywords:
                    # Get patient tests from wide table
                    tests = self.db.get_patient_tests(keyword)
                    if tests:
                        # Get patient info
                        reports = self.db.get_medical_reports()
                        patient_report = None
                        for report in reports:
                            if keyword.lower() in report['patient_name'].lower():
                                patient_report = report
                                break
                        
                        if patient_report:
                            context = f"Patient: {patient_report['patient_name']} ({patient_report['patient_age']}, {patient_report['patient_sex']})"
                            context += f"\nCollection Date: {patient_report['collection_date']}"
                            context += f"\nReport Date: {patient_report['report_date']}"
                            
                            # Add test results
                            for test in tests:
                                test_info = f"\n{test['test_name']}: {test['test_value']} {test['test_unit']}"
                                if test['reference_range']:
                                    test_info += f" (Reference: {test['reference_range']})"
                                if test['is_abnormal']:
                                    test_info += " [ABNORMAL]"
                                context += test_info
                            
                            contexts.append(context)
                            sources.append(f"Medical Report - {patient_report['patient_name']}")
            
            # Search for specific medical tests
            test_keywords = self._extract_test_keywords(query_lower)
            if test_keywords:
                reports = self.db.get_medical_reports()
                for report in reports:
                    tests = self.db.get_patient_tests(report['patient_name'])
                    relevant_tests = []
                    
                    for test in tests:
                        test_name = test['test_name'].lower()
                        if any(keyword in test_name for keyword in test_keywords):
                            relevant_tests.append(test)
                    
                    if relevant_tests:
                        context = f"Patient: {report['patient_name']} - Relevant Tests:"
                        for test in relevant_tests:
                            context += f"\n{test['test_name']}: {test['test_value']} {test['test_unit']}"
                            if test['reference_range']:
                                context += f" (Reference: {test['reference_range']})"
                        
                        contexts.append(context)
                        sources.append(f"Medical Tests - {report['patient_name']}")
            
            # If no specific results, get all available medical data
            if not contexts:
                reports = self.db.get_medical_reports()
                for report in reports:
                    tests = self.db.get_patient_tests(report['patient_name'])
                    if tests:
                        context = f"Patient: {report['patient_name']} ({report['patient_age']}, {report['patient_sex']})"
                        context += f"\nTotal Tests: {len(tests)}"
                        # Show first few tests
                        for test in tests[:5]:
                            context += f"\n{test['test_name']}: {test['test_value']} {test['test_unit']}"
                        
                        contexts.append(context)
                        sources.append(f"Medical Report - {report['patient_name']}")
            
            return {
                'contexts': contexts,
                'sources': sources
            }
            
        except Exception as e:
            logger.error(f"Error searching medical data: {str(e)}")
            return {'contexts': [], 'sources': []}
    
    def _extract_patient_keywords(self, query: str) -> List[str]:
        """Extract patient name keywords from query"""
        try:
            # Get all patient names from database
            reports = self.db.get_medical_reports()
            patient_names = []
            for report in reports:
                if report['patient_name']:
                    name_parts = report['patient_name'].lower().split()
                    patient_names.extend(name_parts)
            
            # Add common patient-related terms
            patient_names.extend(['patient', 'report', 'medical'])
            patient_names = list(set(patient_names))
            
            # Find keywords in query
            found_keywords = []
            for name in patient_names:
                if name in query:
                    found_keywords.append(name)
            
            return found_keywords
        except Exception as e:
            logger.error(f"Error extracting patient keywords: {str(e)}")
            return ['patient', 'report']
    
    def _extract_test_keywords(self, query: str) -> List[str]:
        """Extract medical test keywords from query"""
        medical_test_keywords = [
            # Biochemistry
            'glucose', 'fbs', 'rbs', 'blood sugar', 'bun', 'urea', 'creatinine', 'uric acid', 'phosphorus',
            
            # Liver Function
            'bilirubin', 'sgot', 'sgpt', 'ast', 'alt', 'ggt', 'alkaline phosphatase', 
            'protein', 'albumin', 'globulin',
            
            # Lipid Profile
            'cholesterol', 'triglycerides', 'hdl', 'ldl', 'vldl', 'lipid',
            
            # Thyroid
            't3', 't4', 'tsh', 'thyroid',
            
            # Immunoassay
            'vitamin d', 'vitamin'
        ]
        
        found_keywords = []
        for keyword in medical_test_keywords:
            if keyword in query:
                found_keywords.append(keyword)
        
        return found_keywords


def create_medical_prompt(user_query: str, medical_results: Dict[str, Any]) -> str:
    """Create an enhanced prompt for medical queries"""
    contexts = medical_results.get('contexts', [])
    context_str = "\n\n".join(contexts) if contexts else "No specific medical data found."
    
    return f"""You are an intelligent medical assistant AI. You have access to medical lab reports and patient data. 
Provide helpful, accurate, and empathetic responses about medical information.

IMPORTANT GUIDELINES:
- Always remind users that this is for informational purposes only and not a substitute for professional medical advice
- If abnormal values are found, suggest consulting with a healthcare provider
- Explain medical terms in simple language
- Be encouraging and supportive in your tone
- Use the provided medical data to give specific, relevant answers

Available Medical Context:
{context_str}

User Question: {user_query}

Please provide a helpful response based on the available medical data. If you identify any abnormal values, explain what they might indicate and recommend consulting a healthcare professional."""

