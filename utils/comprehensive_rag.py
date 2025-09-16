"""
Comprehensive RAG system that connects to all database tables
"""

import logging
from typing import List, Dict, Any, Optional
from models.database import DatabaseManager
from sqlalchemy import text

# Setup logging
logger = logging.getLogger(__name__)

class ComprehensiveRAG:
    """Comprehensive RAG system for all database tables"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def search_all_data(self, query: str) -> Dict[str, Any]:
        """Search across all database tables based on query"""
        try:
            contexts = []
            sources = []
            query_lower = query.lower()
            
            # Medical data search
            medical_context = self._search_medical_data(query_lower)
            if medical_context['contexts']:
                contexts.extend(medical_context['contexts'])
                sources.extend(medical_context['sources'])
            
            # Document data search
            document_context = self._search_document_data(query_lower)
            if document_context['contexts']:
                contexts.extend(document_context['contexts'])
                sources.extend(document_context['sources'])
            
            # Chat history search
            chat_context = self._search_chat_data(query_lower)
            if chat_context['contexts']:
                contexts.extend(chat_context['contexts'])
                sources.extend(chat_context['sources'])
            
            # User data search
            user_context = self._search_user_data(query_lower)
            if user_context['contexts']:
                contexts.extend(user_context['contexts'])
                sources.extend(user_context['sources'])
            
            # Job data search
            job_context = self._search_job_data(query_lower)
            if job_context['contexts']:
                contexts.extend(job_context['contexts'])
                sources.extend(job_context['sources'])
            
            # Optimize context selection to reduce token usage
            optimized_contexts = self._optimize_context_selection(contexts, max_contexts=5)
            
            return {
                'contexts': optimized_contexts,
                'sources': sources,
                'query_type': self._determine_query_type(query_lower),
                'total_results': len(contexts),
                'optimized_results': len(optimized_contexts)
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive search: {str(e)}")
            return {'contexts': [], 'sources': [], 'query_type': 'general', 'total_results': 0}
    
    def _optimize_context_selection(self, contexts: List[str], max_contexts: int = 5) -> List[str]:
        """Optimize context selection to reduce token usage while maintaining comprehensive coverage"""
        if len(contexts) <= max_contexts:
            return contexts
        
        # Categorize contexts by test type
        categorized = self._categorize_contexts(contexts)
        
        # Select representative contexts from each category
        optimized = []
        for category, category_contexts in categorized.items():
            if category_contexts:
                # Take the most comprehensive context from each category
                best_context = max(category_contexts, key=len)
                optimized.append(best_context)
        
        # If we still have too many, prioritize by length and diversity
        if len(optimized) > max_contexts:
            # Sort by length (most comprehensive first) and take top N
            optimized = sorted(optimized, key=len, reverse=True)[:max_contexts]
        
        return optimized
    
    def _categorize_contexts(self, contexts: List[str]) -> Dict[str, List[str]]:
        """Categorize contexts by test type for better coverage"""
        categories = {
            'biochemistry': [],
            'thyroid': [],
            'lipid': [],
            'liver': [],
            'kidney': [],
            'other': []
        }
        
        for context in contexts:
            context_lower = context.lower()
            
            # Categorize based on content
            if any(test in context_lower for test in ['glucose', 'rbs', 'fbs', 'hba1c', 'diabetes']):
                categories['biochemistry'].append(context)
            elif any(test in context_lower for test in ['tsh', 't3', 't4', 'thyroid', 'thyroxine']):
                categories['thyroid'].append(context)
            elif any(test in context_lower for test in ['cholesterol', 'hdl', 'ldl', 'triglycerides', 'lipid']):
                categories['lipid'].append(context)
            elif any(test in context_lower for test in ['sgot', 'sgpt', 'ast', 'alt', 'liver', 'bilirubin']):
                categories['liver'].append(context)
            elif any(test in context_lower for test in ['creatinine', 'bun', 'urea', 'kidney', 'renal']):
                categories['kidney'].append(context)
            else:
                categories['other'].append(context)
        
        return categories
    
    def create_comprehensive_test_summary(self, query: str) -> str:
        """Create a comprehensive test summary covering all test types"""
        try:
            # Get all medical data
            medical_results = self._search_medical_data(query.lower())
            
            if not medical_results['contexts']:
                return "No medical data available for analysis."
            
            # Create a structured summary covering all test types
            summary_parts = []
            
            # Group by patient and test type
            patients = {}
            for context in medical_results['contexts']:
                # Extract patient name from context
                lines = context.split('\\n')
                patient_name = None
                for line in lines:
                    if 'Medical Report for' in line:
                        patient_name = line.split('Medical Report for')[1].split(':')[0].strip()
                        break
                
                if patient_name:
                    if patient_name not in patients:
                        patients[patient_name] = []
                    patients[patient_name].append(context)
            
            # Create comprehensive summary
            for patient_name, patient_contexts in patients.items():
                summary_parts.append(f"\\n=== {patient_name} - COMPREHENSIVE TEST ANALYSIS ===")
                
                # Combine all contexts for this patient
                combined_context = '\\n'.join(patient_contexts)
                
                # Extract key test categories
                test_categories = self._extract_test_categories(combined_context)
                
                for category, tests in test_categories.items():
                    if tests:
                        summary_parts.append(f"\\n{category.upper()}:")
                        summary_parts.extend(tests)
            
            return '\\n'.join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error creating comprehensive test summary: {str(e)}")
            return "Error creating test summary."
    
    def _extract_test_categories(self, context: str) -> Dict[str, List[str]]:
        """Extract test categories from medical context"""
        categories = {
            'biochemistry': [],
            'thyroid': [],
            'lipid': [],
            'liver': [],
            'kidney': [],
            'protein': [],
            'vitamin': []
        }
        
        lines = context.split('\\n')
        current_category = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for category headers
            if 'biochemistry:' in line.lower():
                current_category = 'biochemistry'
            elif 'thyroid' in line.lower() and ':' in line:
                current_category = 'thyroid'
            elif 'lipid' in line.lower() and ':' in line:
                current_category = 'lipid'
            elif 'liver' in line.lower() and ':' in line:
                current_category = 'liver'
            elif 'kidney' in line.lower() and ':' in line:
                current_category = 'kidney'
            elif 'protein' in line.lower() and ':' in line:
                current_category = 'protein'
            elif 'vitamin' in line.lower() and ':' in line:
                current_category = 'vitamin'
            elif line.startswith('-') and current_category:
                # This is a test result
                categories[current_category].append(line)
        
        return categories
    
    def _search_medical_data(self, query: str) -> Dict[str, Any]:
        """Search medical data from all medical tables"""
        try:
            contexts = []
            sources = []
            
            # Create a new connection for this search
            conn = self.db.engine.connect()
            try:
                # Search medical_reports_wide table for comprehensive test results
                result = conn.execute(text("""
                    SELECT patient_name, patient_age, patient_sex, patient_id,
                           collection_date, report_date, created_at,
                           glucose_rbs, bun, urea, creatinine, uric_acid, phosphorus,
                           bilirubin_total, bilirubin_direct, bilirubin_indirect,
                           sgot_ast, sgpt_alt, ggt, alkaline_phosphatase,
                           total_protein, albumin, globulin, ag_ratio,
                           cholesterol_total, triglycerides, hdl_cholesterol, 
                           ldl_cholesterol, vldl_cholesterol, non_hdl_cholesterol,
                           total_chol_hdl_ratio, trig_hdl_ratio, ldl_hdl_ratio,
                           vitamin_d, t3_total, t4_total, tsh
                    FROM medical_reports_wide 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """))
                
                reports = result.fetchall()
                for report in reports:
                    # Extract patient info
                    patient_name = report[0] or "Unknown"
                    patient_age = report[1] or "Unknown"
                    patient_sex = report[2] or "Unknown"
                    patient_id = report[3] or "Unknown"
                    collection_date = report[4] or "Unknown"
                    report_date = report[5] or "Unknown"
                    
                    # Build comprehensive test results context with ALL available data
                    test_context = f"""
Medical Report for {patient_name}:
- Patient ID: {patient_id}
- Age: {patient_age} years, Sex: {patient_sex}
- Collection Date: {collection_date}
- Report Date: {report_date}

COMPREHENSIVE TEST RESULTS:

Biochemistry:
- Glucose (RBS): {report[7] or 'N/A'} mg/dL
- Blood Urea Nitrogen (BUN): {report[8] or 'N/A'} mg/dL
- Urea: {report[9] or 'N/A'} mg/dL
- Creatinine: {report[10] or 'N/A'} mg/dL
- Uric Acid: {report[11] or 'N/A'} mg/dL
- Phosphorus: {report[12] or 'N/A'} mg/dL

Liver Function Tests:
- Bilirubin Total: {report[13] or 'N/A'} mg/dL
- Bilirubin Direct: {report[14] or 'N/A'} mg/dL
- Bilirubin Indirect: {report[15] or 'N/A'} mg/dL
- SGOT/AST: {report[16] or 'N/A'} U/L
- SGPT/ALT: {report[17] or 'N/A'} U/L
- GGT: {report[18] or 'N/A'} U/L
- Alkaline Phosphatase: {report[19] or 'N/A'} U/L

Protein Profile:
- Total Protein: {report[20] or 'N/A'} g/dL
- Albumin: {report[21] or 'N/A'} g/dL
- Globulin: {report[22] or 'N/A'} g/dL
- A/G Ratio: {report[23] or 'N/A'}

Lipid Profile:
- Total Cholesterol: {report[24] or 'N/A'} mg/dL
- Triglycerides: {report[25] or 'N/A'} mg/dL
- HDL Cholesterol: {report[26] or 'N/A'} mg/dL
- LDL Cholesterol: {report[27] or 'N/A'} mg/dL
- VLDL Cholesterol: {report[28] or 'N/A'} mg/dL
- Non-HDL Cholesterol: {report[29] or 'N/A'} mg/dL
- Total Chol/HDL Ratio: {report[30] or 'N/A'}
- Triglyceride/HDL Ratio: {report[31] or 'N/A'}
- LDL/HDL Ratio: {report[32] or 'N/A'}

Thyroid Profile:
- T3 (Total): {report[33] or 'N/A'} ng/dL
- T4 (Total): {report[34] or 'N/A'} μg/dL
- TSH: {report[35] or 'N/A'} μIU/mL

Immunoassay:
- Vitamin D (25-Hydroxy): {report[36] or 'N/A'} ng/mL
"""
                    contexts.append(test_context.strip())
                    sources.append(f"Complete Medical Report - {patient_name}")
                
            finally:
                conn.close()
            
            return {'contexts': contexts, 'sources': sources}
            
        except Exception as e:
            logger.error(f"Error searching medical data: {str(e)}")
            return {'contexts': [], 'sources': []}
    
    def _search_document_data(self, query: str) -> Dict[str, Any]:
        """Search document data"""
        try:
            contexts = []
            sources = []
            
            conn = self.db.engine.connect()
            try:
                # Search documents table
                result = conn.execute(text("""
                    SELECT filename, content, created_at
                    FROM documents 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """))
                
                documents = result.fetchall()
                for doc in documents:
                    content_preview = doc[1][:200] + "..." if len(doc[1]) > 200 else doc[1]
                    context = f"""
Document: {doc[0]}
Content Preview: {content_preview}
Uploaded: {doc[2]}
"""
                    contexts.append(context.strip())
                    sources.append(f"Document - {doc[0]}")
            finally:
                conn.close()
            
            return {'contexts': contexts, 'sources': sources}
            
        except Exception as e:
            logger.error(f"Error searching document data: {str(e)}")
            return {'contexts': [], 'sources': []}
    
    def _search_chat_data(self, query: str) -> Dict[str, Any]:
        """Search chat history"""
        try:
            contexts = []
            sources = []
            
            conn = self.db.engine.connect()
            try:
                # Search recent chat messages
                result = conn.execute(text("""
                    SELECT role, content, created_at
                    FROM chat_messages 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """))
                
                messages = result.fetchall()
                if messages:
                    chat_context = "Recent Chat History:\n"
                    for msg in messages:
                        chat_context += f"[{msg[0]}] {msg[2]}: {msg[1][:100]}...\n"
                    
                    contexts.append(chat_context.strip())
                    sources.append("Chat History")
            finally:
                conn.close()
            
            return {'contexts': contexts, 'sources': sources}
            
        except Exception as e:
            logger.error(f"Error searching chat data: {str(e)}")
            return {'contexts': [], 'sources': []}
    
    def _search_user_data(self, query: str) -> Dict[str, Any]:
        """Search user data"""
        try:
            contexts = []
            sources = []
            
            with self.db.engine.connect() as conn:
                # Search users table
                result = conn.execute(text("""
                    SELECT name, email, role, created_at
                    FROM users 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """))
                
                users = result.fetchall()
                if users:
                    user_context = "System Users:\n"
                    for user in users:
                        user_context += f"- {user[0]} ({user[1]}) - Role: {user[2]}\n"
                    
                    contexts.append(user_context.strip())
                    sources.append("User Database")
            
            return {'contexts': contexts, 'sources': sources}
            
        except Exception as e:
            logger.error(f"Error searching user data: {str(e)}")
            return {'contexts': [], 'sources': []}
    
    def _search_job_data(self, query: str) -> Dict[str, Any]:
        """Search job-related data"""
        try:
            contexts = []
            sources = []
            
            with self.db.engine.connect() as conn:
                # Search jobs table
                result = conn.execute(text("""
                    SELECT title, description, location, salary_range, job_type
                    FROM jobs 
                    WHERE is_active = true
                    ORDER BY created_at DESC 
                    LIMIT 5
                """))
                
                jobs = result.fetchall()
                if jobs:
                    job_context = "Available Jobs:\n"
                    for job in jobs:
                        job_context += f"- {job[0]} at {job[2]}\n  Salary: {job[3]}\n  Type: {job[4]}\n"
                    
                    contexts.append(job_context.strip())
                    sources.append("Job Database")
            
            return {'contexts': contexts, 'sources': sources}
            
        except Exception as e:
            logger.error(f"Error searching job data: {str(e)}")
            return {'contexts': [], 'sources': []}
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query"""
        if any(word in query for word in ['medical', 'patient', 'test', 'health', 'doctor']):
            return 'medical'
        elif any(word in query for word in ['document', 'file', 'upload', 'pdf']):
            return 'document'
        elif any(word in query for word in ['chat', 'message', 'conversation']):
            return 'chat'
        elif any(word in query for word in ['user', 'admin', 'role', 'permission']):
            return 'user'
        elif any(word in query for word in ['job', 'career', 'position', 'employment']):
            return 'job'
        else:
            return 'general'
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            stats = {}
            
            with self.db.engine.connect() as conn:
                # Get table counts
                tables = [
                    'medical_reports', 'medical_tests', 'documents', 
                    'chat_messages', 'users', 'jobs', 'candidates'
                ]
                
                for table in tables:
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.scalar()
                        stats[table] = count
                    except:
                        stats[table] = 0
                
                # Get recent activity
                result = conn.execute(text("""
                    SELECT 
                        (SELECT COUNT(*) FROM medical_reports WHERE created_at > NOW() - INTERVAL '7 days') as recent_reports,
                        (SELECT COUNT(*) FROM chat_messages WHERE created_at > NOW() - INTERVAL '7 days') as recent_messages,
                        (SELECT COUNT(*) FROM documents WHERE created_at > NOW() - INTERVAL '7 days') as recent_documents
                """))
                
                activity = result.fetchone()
                stats['recent_activity'] = {
                    'reports_7_days': activity[0] if activity else 0,
                    'messages_7_days': activity[1] if activity else 0,
                    'documents_7_days': activity[2] if activity else 0
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {}
