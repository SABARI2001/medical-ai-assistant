from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import logging
import json
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create base class for declarative models
Base = declarative_base()

class Document(Base):
    """Model for storing processed documents"""
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    chunks = relationship("DocumentChunk", back_populates="document")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DocumentChunk(Base):
    """Model for storing document chunks for RAG"""
    __tablename__ = 'document_chunks'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'))
    content = Column(Text, nullable=False)
    embedding = Column(Text)  # Store embeddings as serialized numpy array
    document = relationship("Document", back_populates="chunks")
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatMessage(Base):
    """Model for storing chat messages"""
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True)
    role = Column(String(50), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    conversation_id = Column(String(100))  # Group messages by conversation
    created_at = Column(DateTime, default=datetime.utcnow)
    sources = relationship("MessageSource", back_populates="message")

class MessageSource(Base):
    """Model for storing message sources (for citations)"""
    __tablename__ = 'message_sources'
    
    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey('chat_messages.id'))
    source_type = Column(String(50))  # 'document', 'web', etc.
    source_id = Column(String(255))  # Document ID, URL, etc.
    content = Column(Text)  # Relevant excerpt or description
    message = relationship("ChatMessage", back_populates="sources")
    created_at = Column(DateTime, default=datetime.utcnow)

class MedicalReport(Base):
    """Model for storing extracted medical report data"""
    __tablename__ = 'medical_reports'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'))
    patient_name = Column(String(255))
    patient_age = Column(String(10))
    patient_sex = Column(String(10))
    patient_id = Column(String(100))
    collection_date = Column(String(50))
    report_date = Column(String(50))
    extracted_data = Column(JSON)  # Store the full JSON structure
    document = relationship("Document")
    tests = relationship("MedicalTest", back_populates="report")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class MedicalTest(Base):
    """Model for storing individual medical test results with specific test columns"""
    __tablename__ = 'medical_tests'
    
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey('medical_reports.id'))
    
    # Patient Information
    patient_name = Column(String(255))
    patient_age = Column(String(10))
    patient_sex = Column(String(10))
    
    # Test Category and General Info
    test_category = Column(String(50))  # biochemistry, liverFunction, etc.
    test_name = Column(String(255))
    test_value = Column(String(50))
    test_unit = Column(String(20))
    reference_range = Column(String(100))
    is_abnormal = Column(Boolean, default=False)
    
    # Biochemistry Tests
    glucose_fasting_fbs = Column(Float)
    glucose_postprandial_ppbs = Column(Float)
    blood_urea_nitrogen_bun = Column(Float)
    urea = Column(Float)
    creatinine = Column(Float)
    uric_acid = Column(Float)
    phosphorus = Column(Float)
    
    # Liver Function Tests
    bilirubin_total = Column(Float)
    bilirubin_direct = Column(Float)
    bilirubin_indirect = Column(Float)
    sgot_ast = Column(Float)  # Aspartate Aminotransferase
    sgpt_alt = Column(Float)  # Alanine Aminotransferase
    ggt = Column(Float)  # Gamma Glutamyl Transpeptidase
    alkaline_phosphatase_sap = Column(Float)
    total_protein = Column(Float)
    albumin = Column(Float)
    globulin = Column(Float)
    ag_ratio = Column(Float)
    
    # Lipid Profile
    cholesterol_total = Column(Float)
    triglycerides = Column(Float)
    hdl_cholesterol = Column(Float)
    ldl_cholesterol = Column(Float)
    vldl_cholesterol = Column(Float)
    non_hdl_cholesterol = Column(Float)
    total_chol_hdl_ratio = Column(Float)
    triglyceride_hdl_ratio = Column(Float)
    ldl_hdl_ratio = Column(Float)
    
    # Diabetes Markers
    glycosylated_hemoglobin_hba1c = Column(Float)
    estimated_average_glucose = Column(Float)
    
    # Immunoassay
    vitamin_d_25_hydroxy = Column(Float)
    
    # Thyroid Profile (TFT)
    t3_triiodothyronine_total = Column(Float)
    t4_thyroxine_total = Column(Float)
    tsh_thyroid_stimulating_hormone = Column(Float)
    
    # Additional fields for backward compatibility
    glucose_random_rbs = Column(Float)
    
    report = relationship("MedicalReport", back_populates="tests")
    created_at = Column(DateTime, default=datetime.utcnow)

class MedicalReportWide(Base):
    """Wide table format for storing all medical test results - optimized for RAG queries"""
    __tablename__ = "medical_reports_wide"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_name = Column(String(255))
    patient_age = Column(String(10))
    patient_sex = Column(String(10))
    patient_id = Column(String(100))
    collection_date = Column(String(50))
    report_date = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    # -----------------------------
    # Biochemistry
    glucose_rbs = Column(Float)
    bun = Column(Float)          # Blood Urea Nitrogen
    urea = Column(Float)
    creatinine = Column(Float)
    uric_acid = Column(Float)
    phosphorus = Column(Float)

    # -----------------------------
    # Liver Function Tests
    bilirubin_total = Column(Float)
    bilirubin_direct = Column(Float)
    bilirubin_indirect = Column(Float)
    sgot_ast = Column(Float)
    sgpt_alt = Column(Float)
    ggt = Column(Float)
    alkaline_phosphatase = Column(Float)
    total_protein = Column(Float)
    albumin = Column(Float)
    globulin = Column(Float)
    ag_ratio = Column(Float)

    # -----------------------------
    # Lipid Profile
    cholesterol_total = Column(Float)
    triglycerides = Column(Float)
    hdl_cholesterol = Column(Float)
    ldl_cholesterol = Column(Float)
    vldl_cholesterol = Column(Float)
    non_hdl_cholesterol = Column(Float)
    total_chol_hdl_ratio = Column(Float)
    trig_hdl_ratio = Column(Float)
    ldl_hdl_ratio = Column(Float)

    # -----------------------------
    # Immunoassay
    vitamin_d = Column(Float)

    # -----------------------------
    # Thyroid Profile
    t3_total = Column(Float)
    t4_total = Column(Float)
    tsh = Column(Float)

class DatabaseManager:
    """Manager class for database operations"""
    
    def __init__(self, host='127.0.0.1', port=5432, dbname='postgres', user='postgres', password='1234'):
        self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = None
        self.Session = None
        self.setup_database()
    
    def setup_database(self):
        """Set up database connection and create tables"""
        try:
            self.engine = create_engine(self.connection_string)
            
            # Create tables if they don't exist (don't drop existing ones)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up database: {str(e)}")
            raise
    
    def store_document(self, filename: str, content: str, chunks: list = None):
        """Store a document and its chunks"""
        session = self.Session()
        try:
            document = Document(filename=filename, content=content)
            session.add(document)
            
            if chunks:
                for chunk_content in chunks:
                    chunk = DocumentChunk(document=document, content=chunk_content)
                    session.add(chunk)
            
            session.commit()
            logger.info(f"Stored document: {filename}")
            return document.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing document: {str(e)}")
            raise
        finally:
            session.close()
    
    def store_chat_message(self, role: str, content: str, conversation_id: str, sources: list = None):
        """Store a chat message and its sources"""
        session = self.Session()
        try:
            message = ChatMessage(
                role=role,
                content=content,
                conversation_id=conversation_id
            )
            session.add(message)
            
            if sources:
                for source in sources:
                    # Handle both string and dict sources
                    if isinstance(source, str):
                        msg_source = MessageSource(
                            message=message,
                            source_type='text',
                            source_id=None,
                            content=source
                        )
                    else:
                        msg_source = MessageSource(
                            message=message,
                            source_type=source.get('type'),
                            source_id=source.get('id'),
                            content=source.get('content')
                        )
                    session.add(msg_source)
            
            session.commit()
            logger.info(f"Stored chat message for conversation: {conversation_id}")
            return message.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing chat message: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_conversation_history(self, conversation_id: str, limit: int = None):
        """Get chat history for a conversation"""
        session = self.Session()
        try:
            query = session.query(ChatMessage).filter_by(conversation_id=conversation_id)\
                .order_by(ChatMessage.created_at)
            
            if limit:
                query = query.limit(limit)
            
            messages = query.all()
            
            # Convert to dictionary format
            history = []
            for msg in messages:
                message_dict = {
                    'role': msg.role,
                    'content': msg.content,
                    'created_at': msg.created_at,
                    'sources': []
                }
                
                for source in msg.sources:
                    message_dict['sources'].append({
                        'type': source.source_type,
                        'id': source.source_id,
                        'content': source.content
                    })
                
                history.append(message_dict)
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_document_chunks(self, document_id: int = None):
        """Get all chunks for a document or all chunks if no document_id provided"""
        session = self.Session()
        try:
            query = session.query(DocumentChunk)
            if document_id:
                query = query.filter_by(document_id=document_id)
            
            chunks = query.all()
            return [(chunk.content, chunk.embedding) for chunk in chunks]
        except Exception as e:
            logger.error(f"Error retrieving document chunks: {str(e)}")
            raise
        finally:
            session.close()
    
    def store_medical_report(self, document_id: int, extracted_data: dict):
        """Store extracted medical report data with specific test columns"""
        session = self.Session()
        try:
            patient_info = extracted_data.get('patient', {})
            
            # Create medical report record
            medical_report = MedicalReport(
                document_id=document_id,
                patient_name=patient_info.get('name', ''),
                patient_age=patient_info.get('age', ''),
                patient_sex=patient_info.get('sex', ''),
                patient_id=patient_info.get('patientId', ''),
                collection_date=patient_info.get('collectionDate', ''),
                report_date=patient_info.get('reportDate', ''),
                extracted_data=extracted_data
            )
            session.add(medical_report)
            session.flush()  # Get the ID
            
            # Create a comprehensive medical test record with all specific columns
            medical_test = MedicalTest(
                report_id=medical_report.id,
                patient_name=patient_info.get('name', ''),
                patient_age=patient_info.get('age', ''),
                patient_sex=patient_info.get('sex', ''),
                test_category='comprehensive',
                test_name='Complete Medical Panel',
                test_value='',
                test_unit='',
                reference_range='',
                is_abnormal=False
            )
            
            # Map extracted data to specific test columns
            self._map_tests_to_columns(medical_test, extracted_data)
            
            session.add(medical_test)
            
            # ALSO store in wide table format for RAG queries
            wide_record = self._create_wide_record(extracted_data)
            session.add(wide_record)
            
            session.commit()
            logger.info(f"Stored medical report for patient: {patient_info.get('name', 'Unknown')}")
            return medical_report.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing medical report: {str(e)}")
            raise
        finally:
            session.close()
    
    def _create_wide_record(self, extracted_data: dict):
        """Create a wide table record from extracted data"""
        patient_info = extracted_data.get('patient', {})
        
        # Initialize the wide record
        wide_record = MedicalReportWide(
            patient_name=patient_info.get('name', ''),
            patient_age=patient_info.get('age', ''),
            patient_sex=patient_info.get('sex', ''),
            patient_id=patient_info.get('patientId', ''),
            collection_date=patient_info.get('collectionDate', ''),
            report_date=patient_info.get('reportDate', '')
        )
        
        # Map test results to wide table columns
        test_mappings = {
            # Biochemistry
            'glucose_rbs': ['Glucose - Random (RBS)', 'Glucose Fasting', 'Glucose'],
            'bun': ['Blood Urea Nitrogen (BUN)', 'BUN'],
            'urea': ['Urea'],
            'creatinine': ['Creatinine'],
            'uric_acid': ['Uric Acid'],
            'phosphorus': ['Phosphorus'],
            
            # Liver Function Tests
            'bilirubin_total': ['Bilirubin Total', 'Bilirubin (Total)'],
            'bilirubin_direct': ['Bilirubin Direct', 'Bilirubin (Direct)'],
            'bilirubin_indirect': ['Bilirubin Indirect', 'Bilirubin (Indirect)'],
            'sgot_ast': ['SGOT/AST', 'AST', 'Aspartate Aminotransferase'],
            'sgpt_alt': ['SGPT/ALT', 'ALT', 'Alanine Aminotransferase'],
            'ggt': ['GGT', 'Gamma Glutamyl Transpeptidase'],
            'alkaline_phosphatase': ['Alkaline Phosphatase', 'SAP'],
            'total_protein': ['Total Protein'],
            'albumin': ['Albumin'],
            'globulin': ['Globulin'],
            'ag_ratio': ['A/G Ratio'],
            
            # Lipid Profile
            'cholesterol_total': ['Cholesterol Total', 'Cholesterol (Total)'],
            'triglycerides': ['Triglycerides'],
            'hdl_cholesterol': ['HDL Cholesterol'],
            'ldl_cholesterol': ['LDL Cholesterol'],
            'vldl_cholesterol': ['VLDL Cholesterol'],
            'non_hdl_cholesterol': ['Non-HDL Cholesterol'],
            'total_chol_hdl_ratio': ['Total Chol/HDL Ratio'],
            'trig_hdl_ratio': ['Triglyceride/HDL Ratio'],
            'ldl_hdl_ratio': ['LDL/HDL Ratio'],
            
            # Immunoassay
            'vitamin_d': ['Vitamin D', 'Vitamin D (25-Hydroxy Vit D)'],
            
            # Thyroid Profile
            't3_total': ['T3 (Triiodothyronine) - Total', 'T3 Total'],
            't4_total': ['T4 (Thyroxine) - Total', 'T4 Total'],
            'tsh': ['TSH (Thyroid Stimulating Hormone)', 'TSH']
        }
        
        # Process all test categories
        for category in ['biochemistry', 'liverFunction', 'lipidProfile', 'thyroidProfile', 'immunoassay', 'other']:
            tests = extracted_data.get(category, [])
            for test in tests:
                test_name = test.get('test', '').strip()
                test_value = test.get('value', '').strip()
                
                if not test_name or not test_value:
                    continue
                
                # Try to convert value to float
                try:
                    numeric_value = float(test_value)
                except (ValueError, TypeError):
                    continue
                
                # Map test to wide table column
                for column_name, test_patterns in test_mappings.items():
                    if any(pattern.lower() in test_name.lower() for pattern in test_patterns):
                        setattr(wide_record, column_name, numeric_value)
                        logger.info(f"Mapped {test_name}: {test_value} to {column_name}")
                        break
        
        return wide_record
    
    def _map_tests_to_columns(self, medical_test, extracted_data):
        """Map extracted test data to specific database columns"""
        # Test name mappings to column names with reference ranges
        test_mappings = {
            # Biochemistry
            'glucose_fasting_fbs': ['Glucose Fasting (FBS)', 'Glucose Fasting', 'FBS'],
            'glucose_postprandial_ppbs': ['Glucose Postprandial (PPBS)', 'Glucose Postprandial', 'PPBS'],
            'glucose_random_rbs': ['Glucose - Random (RBS)', 'Glucose Random', 'RBS', 'Glucose'],
            'blood_urea_nitrogen_bun': ['Blood Urea Nitrogen (BUN)', 'BUN'],
            'urea': ['Urea'],
            'creatinine': ['Creatinine'],
            'uric_acid': ['Uric Acid'],
            'phosphorus': ['Phosphorus'],
            
            # Liver Function Tests
            'bilirubin_total': ['Bilirubin (Total)', 'Bilirubin Total'],
            'bilirubin_direct': ['Bilirubin (Direct)', 'Bilirubin Direct'],
            'bilirubin_indirect': ['Bilirubin (Indirect)', 'Bilirubin Indirect'],
            'sgot_ast': ['SGOT/AST (Aspartate Aminotransferase)', 'SGOT/AST', 'AST'],
            'sgpt_alt': ['SGPT/ALT (Alanine Aminotransferase)', 'SGPT/ALT', 'ALT'],
            'ggt': ['GGT (Gamma Glutamyl Transpeptidase)', 'GGT'],
            'alkaline_phosphatase_sap': ['Alkaline Phosphatase (SAP)', 'Alkaline Phosphatase', 'SAP'],
            'total_protein': ['Total Protein'],
            'albumin': ['Albumin'],
            'globulin': ['Globulin'],
            'ag_ratio': ['A/G Ratio'],
            
            # Lipid Profile
            'cholesterol_total': ['Cholesterol (Total)', 'Cholesterol Total'],
            'triglycerides': ['Triglycerides'],
            'hdl_cholesterol': ['HDL Cholesterol'],
            'ldl_cholesterol': ['LDL Cholesterol'],
            'vldl_cholesterol': ['VLDL Cholesterol'],
            'non_hdl_cholesterol': ['Non-HDL Cholesterol'],
            'total_chol_hdl_ratio': ['Total Cholesterol/HDL Ratio'],
            'triglyceride_hdl_ratio': ['Triglyceride/HDL Cholesterol Ratio'],
            'ldl_hdl_ratio': ['LDL/HDL Cholesterol Ratio'],
            
            # Diabetes Markers
            'glycosylated_hemoglobin_hba1c': ['Glycosylated Hemoglobin (HbA1c)', 'HbA1c'],
            'estimated_average_glucose': ['Estimated Average Glucose'],
            
            # Immunoassay
            'vitamin_d_25_hydroxy': ['Vitamin D (25-Hydroxy Vit D)', 'Vitamin D', '25-Hydroxy Vit D'],
            
            # Thyroid Profile
            't3_triiodothyronine_total': ['T3 (Triiodothyronine) - Total', 'T3', 'Triiodothyronine'],
            't4_thyroxine_total': ['T4 (Thyroxine) - Total', 'T4', 'Thyroxine'],
            'tsh_thyroid_stimulating_hormone': ['TSH (Thyroid Stimulating Hormone)', 'TSH']
        }
        
        # Collect all tests from all categories
        all_tests = []
        for category in ['biochemistry', 'liverFunction', 'lipidProfile', 'thyroidProfile', 'immunoassay', 'other']:
            tests = extracted_data.get(category, [])
            all_tests.extend(tests)
        
        # Reference ranges for each test
        reference_ranges = {
            'glucose_fasting_fbs': '70-100 mg/dL',
            'glucose_postprandial_ppbs': '<140 mg/dL',
            'glucose_random_rbs': '70-140 mg/dL',
            'blood_urea_nitrogen_bun': '7-20 mg/dL',
            'urea': '15-45 mg/dL',
            'creatinine': '0.6-1.2 mg/dL (Male), 0.5-1.1 mg/dL (Female)',
            'uric_acid': '3.5-7.2 mg/dL (Male), 2.6-6.0 mg/dL (Female)',
            'phosphorus': '2.5-4.5 mg/dL',
            'bilirubin_total': '0.3-1.2 mg/dL',
            'bilirubin_direct': '0.0-0.3 mg/dL',
            'bilirubin_indirect': '0.2-0.9 mg/dL',
            'sgot_ast': '10-40 U/L',
            'sgpt_alt': '10-40 U/L',
            'ggt': '8-38 U/L (Male), 5-27 U/L (Female)',
            'alkaline_phosphatase_sap': '44-147 U/L',
            'total_protein': '6.0-8.3 g/dL',
            'albumin': '3.5-5.0 g/dL',
            'globulin': '2.0-3.5 g/dL',
            'ag_ratio': '1.0-2.0',
            'cholesterol_total': '<200 mg/dL',
            'triglycerides': '<150 mg/dL',
            'hdl_cholesterol': '>40 mg/dL (Male), >50 mg/dL (Female)',
            'ldl_cholesterol': '<100 mg/dL',
            'vldl_cholesterol': '5-40 mg/dL',
            'non_hdl_cholesterol': '<130 mg/dL',
            'total_chol_hdl_ratio': '<5.0',
            'triglyceride_hdl_ratio': '<3.5',
            'ldl_hdl_ratio': '<3.0',
            'glycosylated_hemoglobin_hba1c': '<5.7%',
            'estimated_average_glucose': '70-126 mg/dL',
            'vitamin_d_25_hydroxy': '30-100 ng/mL',
            't3_triiodothyronine_total': '80-200 ng/dL',
            't4_thyroxine_total': '4.5-12.0 μg/dL',
            'tsh_thyroid_stimulating_hormone': '0.4-4.0 μIU/mL'
        }
        
        # Map each test to the appropriate column
        for column_name, test_names in test_mappings.items():
            for test in all_tests:
                test_name = test.get('test', '').strip()
                if any(name.lower() in test_name.lower() or test_name.lower() in name.lower() for name in test_names):
                    try:
                        value = float(test.get('value', 0))
                        setattr(medical_test, column_name, value)
                        
                        # Set reference range
                        if column_name in reference_ranges:
                            medical_test.reference_range = reference_ranges[column_name]
                        
                        logger.info(f"Mapped {test_name}: {value} to {column_name} (Range: {reference_ranges.get(column_name, 'N/A')})")
                        break  # Found a match, move to next column
                    except (ValueError, TypeError):
                        continue
    
    def get_medical_reports(self, patient_name: str = None, patient_id: str = None):
        """Get medical reports with optional filtering"""
        session = self.Session()
        try:
            query = session.query(MedicalReport)
            
            if patient_name:
                query = query.filter(MedicalReport.patient_name.ilike(f'%{patient_name}%'))
            if patient_id:
                query = query.filter_by(patient_id=patient_id)
            
            reports = query.all()
            
            result = []
            for report in reports:
                report_dict = {
                    'id': report.id,
                    'patient_name': report.patient_name,
                    'patient_age': report.patient_age,
                    'patient_sex': report.patient_sex,
                    'patient_id': report.patient_id,
                    'collection_date': report.collection_date,
                    'report_date': report.report_date,
                    'extracted_data': report.extracted_data,
                    'created_at': report.created_at
                }
                result.append(report_dict)
            
            return result
        except Exception as e:
            logger.error(f"Error retrieving medical reports: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_medical_tests(self, report_id: int = None, test_category: str = None, test_name: str = None):
        """Get medical tests with optional filtering"""
        session = self.Session()
        try:
            query = session.query(MedicalTest).join(MedicalReport)
            
            if report_id:
                query = query.filter(MedicalTest.report_id == report_id)
            if test_category:
                query = query.filter(MedicalTest.test_category == test_category)
            if test_name:
                query = query.filter(MedicalTest.test_name.ilike(f'%{test_name}%'))
            
            tests = query.all()
            
            result = []
            for test in tests:
                test_dict = {
                    'id': test.id,
                    'report_id': test.report_id,
                    'patient_name': test.report.patient_name,
                    'test_category': test.test_category,
                    'test_name': test.test_name,
                    'test_value': test.test_value,
                    'test_unit': test.test_unit,
                    'reference_range': test.reference_range,
                    'is_abnormal': test.is_abnormal,
                    'collection_date': test.report.collection_date
                }
                result.append(test_dict)
            
            return result
        except Exception as e:
            logger.error(f"Error retrieving medical tests: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_patient_tests(self, patient_name: str):
        """Get comprehensive test results for a specific patient"""
        session = self.Session()
        try:
            # Query the medical tests table for the patient
            patient_tests = session.query(MedicalTest).join(MedicalReport).filter(
                MedicalTest.patient_name.ilike(f'%{patient_name}%')
            ).first()
            
            if not patient_tests:
                return []
            
            # Convert the wide table format back to individual test format
            tests = []
            
            # Define test definitions with their column mappings
            test_definitions = [
                # Biochemistry
                ('glucose_fasting_fbs', 'Glucose Fasting (FBS)', 'mg/dL'),
                ('glucose_postprandial_ppbs', 'Glucose Postprandial (PPBS)', 'mg/dL'),
                ('glucose_random_rbs', 'Glucose - Random (RBS)', 'mg/dL'),
                ('blood_urea_nitrogen_bun', 'Blood Urea Nitrogen (BUN)', 'mg/dL'),
                ('urea', 'Urea', 'mg/dL'),
                ('creatinine', 'Creatinine', 'mg/dL'),
                ('uric_acid', 'Uric Acid', 'mg/dL'),
                ('phosphorus', 'Phosphorus', 'mg/dL'),
                
                # Liver Function Tests
                ('bilirubin_total', 'Bilirubin (Total)', 'mg/dL'),
                ('bilirubin_direct', 'Bilirubin (Direct)', 'mg/dL'),
                ('bilirubin_indirect', 'Bilirubin (Indirect)', 'mg/dL'),
                ('sgot_ast', 'SGOT/AST (Aspartate Aminotransferase)', 'U/L'),
                ('sgpt_alt', 'SGPT/ALT (Alanine Aminotransferase)', 'U/L'),
                ('ggt', 'GGT (Gamma Glutamyl Transpeptidase)', 'U/L'),
                ('alkaline_phosphatase_sap', 'Alkaline Phosphatase (SAP)', 'U/L'),
                ('total_protein', 'Total Protein', 'g/dL'),
                ('albumin', 'Albumin', 'g/dL'),
                ('globulin', 'Globulin', 'g/dL'),
                ('ag_ratio', 'A/G Ratio', ''),
                
                # Lipid Profile
                ('cholesterol_total', 'Cholesterol (Total)', 'mg/dL'),
                ('triglycerides', 'Triglycerides', 'mg/dL'),
                ('hdl_cholesterol', 'HDL Cholesterol', 'mg/dL'),
                ('ldl_cholesterol', 'LDL Cholesterol', 'mg/dL'),
                ('vldl_cholesterol', 'VLDL Cholesterol', 'mg/dL'),
                ('non_hdl_cholesterol', 'Non-HDL Cholesterol', 'mg/dL'),
                ('total_chol_hdl_ratio', 'Total Cholesterol/HDL Ratio', ''),
                ('triglyceride_hdl_ratio', 'Triglyceride/HDL Cholesterol Ratio', ''),
                ('ldl_hdl_ratio', 'LDL/HDL Cholesterol Ratio', ''),
                
                # Diabetes Markers
                ('glycosylated_hemoglobin_hba1c', 'Glycosylated Hemoglobin (HbA1c)', '%'),
                ('estimated_average_glucose', 'Estimated Average Glucose', 'mg/dL'),
                
                # Immunoassay
                ('vitamin_d_25_hydroxy', 'Vitamin D (25-Hydroxy Vit D)', 'ng/mL'),
                
                # Thyroid Profile
                ('t3_triiodothyronine_total', 'T3 (Triiodothyronine) - Total', 'ng/mL'),
                ('t4_thyroxine_total', 'T4 (Thyroxine) - Total', 'μg/dL'),
                ('tsh_thyroid_stimulating_hormone', 'TSH (Thyroid Stimulating Hormone)', 'μIU/mL')
            ]
            
            # Reference ranges mapping
            reference_ranges = {
                'glucose_fasting_fbs': '70-100 mg/dL',
                'glucose_postprandial_ppbs': '<140 mg/dL',
                'glucose_random_rbs': '70-140 mg/dL',
                'blood_urea_nitrogen_bun': '7-20 mg/dL',
                'urea': '15-45 mg/dL',
                'creatinine': '0.6-1.2 mg/dL (Male), 0.5-1.1 mg/dL (Female)',
                'uric_acid': '3.5-7.2 mg/dL (Male), 2.6-6.0 mg/dL (Female)',
                'phosphorus': '2.5-4.5 mg/dL',
                'bilirubin_total': '0.3-1.2 mg/dL',
                'bilirubin_direct': '0.0-0.3 mg/dL',
                'bilirubin_indirect': '0.2-0.9 mg/dL',
                'sgot_ast': '10-40 U/L',
                'sgpt_alt': '10-40 U/L',
                'ggt': '8-38 U/L (Male), 5-27 U/L (Female)',
                'alkaline_phosphatase_sap': '44-147 U/L',
                'total_protein': '6.0-8.3 g/dL',
                'albumin': '3.5-5.0 g/dL',
                'globulin': '2.0-3.5 g/dL',
                'ag_ratio': '1.0-2.0',
                'cholesterol_total': '<200 mg/dL',
                'triglycerides': '<150 mg/dL',
                'hdl_cholesterol': '>40 mg/dL (Male), >50 mg/dL (Female)',
                'ldl_cholesterol': '<100 mg/dL',
                'vldl_cholesterol': '5-40 mg/dL',
                'non_hdl_cholesterol': '<130 mg/dL',
                'total_chol_hdl_ratio': '<5.0',
                'triglyceride_hdl_ratio': '<3.5',
                'ldl_hdl_ratio': '<3.0',
                'glycosylated_hemoglobin_hba1c': '<5.7%',
                'estimated_average_glucose': '70-126 mg/dL',
                'vitamin_d_25_hydroxy': '30-100 ng/mL',
                't3_triiodothyronine_total': '80-200 ng/dL',
                't4_thyroxine_total': '4.5-12.0 μg/dL',
                'tsh_thyroid_stimulating_hormone': '0.4-4.0 μIU/mL'
            }
            
            for column_name, test_name, unit in test_definitions:
                value = getattr(patient_tests, column_name)
                if value is not None:
                    # Get reference range
                    ref_range = reference_ranges.get(column_name, 'N/A')
                    
                    # Determine if abnormal (basic check)
                    is_abnormal = self._check_abnormal_value(column_name, value, ref_range)
                    
                    tests.append({
                        'test_category': self._get_test_category(test_name),
                        'test_name': test_name,
                        'test_value': value,
                        'test_unit': unit,
                        'reference_range': ref_range,
                        'is_abnormal': is_abnormal
                    })
            
            return tests
        except Exception as e:
            logger.error(f"Error getting patient tests: {str(e)}")
            return []
        finally:
            session.close()
    
    def _get_test_category(self, test_name: str) -> str:
        """Determine test category based on test name"""
        if any(keyword in test_name.lower() for keyword in ['glucose', 'urea', 'creatinine', 'uric', 'phosphorus']):
            return 'biochemistry'
        elif any(keyword in test_name.lower() for keyword in ['bilirubin', 'sgot', 'sgpt', 'ggt', 'alkaline', 'protein', 'albumin', 'globulin']):
            return 'liverFunction'
        elif any(keyword in test_name.lower() for keyword in ['cholesterol', 'triglyceride', 'hdl', 'ldl', 'vldl']):
            return 'lipidProfile'
        elif any(keyword in test_name.lower() for keyword in ['t3', 't4', 'tsh', 'thyroid']):
            return 'thyroidProfile'
        elif any(keyword in test_name.lower() for keyword in ['vitamin d', 'hba1c']):
            return 'immunoassay'
        else:
            return 'other'
    
    def _check_abnormal_value(self, column_name: str, value: float, ref_range: str) -> bool:
        """Check if a test value is abnormal based on reference range"""
        try:
            if not ref_range or ref_range == 'N/A':
                return False
            
            # Parse different reference range formats
            if '<' in ref_range:
                # Handle "< 100" format
                max_val = float(ref_range.replace('<', '').strip().split()[0])
                return value >= max_val
            elif '>' in ref_range:
                # Handle "> 10" format
                min_val = float(ref_range.replace('>', '').strip().split()[0])
                return value <= min_val
            elif '-' in ref_range:
                # Handle "10 - 20" format
                range_parts = ref_range.split('-')
                if len(range_parts) == 2:
                    min_val = float(range_parts[0].strip().split()[0])
                    max_val = float(range_parts[1].strip().split()[0])
                    return value < min_val or value > max_val
            
            return False
        except (ValueError, IndexError):
            return False
    
    def _is_test_abnormal(self, test_data: dict) -> bool:
        """Determine if a test result is abnormal based on reference range"""
        try:
            value_str = test_data.get('value', '').strip()
            range_str = test_data.get('referenceRange', '').strip()
            
            if not value_str or not range_str:
                return False
            
            # Extract numeric value
            value_match = re.search(r'(\d+\.?\d*)', value_str)
            if not value_match:
                return False
            
            value = float(value_match.group(1))
            
            # Parse reference range
            if '<' in range_str:
                # Handle "< 100" format
                max_match = re.search(r'<\s*(\d+\.?\d*)', range_str)
                if max_match:
                    max_val = float(max_match.group(1))
                    return value >= max_val
            elif '>' in range_str:
                # Handle "> 10" format
                min_match = re.search(r'>\s*(\d+\.?\d*)', range_str)
                if min_match:
                    min_val = float(min_match.group(1))
                    return value <= min_val
            elif '-' in range_str:
                # Handle "10 - 20" format
                range_match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', range_str)
                if range_match:
                    min_val = float(range_match.group(1))
                    max_val = float(range_match.group(2))
                    return value < min_val or value > max_val
            
            return False
        except Exception:
            return False

# Test database connection
def test_database_connection():
    """Test database connection and operations"""
    try:
        db = DatabaseManager()
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_database_connection()
