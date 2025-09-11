import streamlit as st
import os
from datetime import datetime
import uuid
import traceback
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Import custom modules
from config.config import Config
from models.llm import LLMManager
from models.embeddings import EmbeddingManager
from models.database import DatabaseManager
from utils.search_utils import WebSearchTool
from utils.medical_extractor_improved import MedicalReportExtractor
from utils.simple_rag_improved import SimpleMedicalRAG, create_medical_prompt
from utils.comprehensive_rag import ComprehensiveRAG
from utils.error_handler import ErrorHandler, retry_on_failure, handle_errors
# API key management is now handled directly in config.py

# Page configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Medical UI Styling
st.markdown("""
<style>
    /* Global Reset and Base Styles */
    * {
        box-sizing: border-box;
    }
    
    /* Force full page scrolling */
    html, body {
        height: auto !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    
    /* Override Streamlit's height constraints */
    .stApp {
        height: auto !important;
        min-height: 100vh !important;
        overflow-y: auto !important;
    }
    
    .stApp > div {
        height: auto !important;
        min-height: 100vh !important;
        overflow-y: visible !important;
    }
    
    /* Main Layout */
    .main {
        height: auto !important;
        min-height: 100vh !important;
        overflow-y: visible !important;
    }
    
    .main .block-container {
        max-width: 1200px;
        padding: 1rem 1.5rem;
        margin: 0 auto;
        background: #fafbfc;
        min-height: 100vh;
        overflow-y: visible !important;
        height: auto !important;
    }
    
    /* Fix content container */
    .main .block-container > div {
        overflow-y: visible !important;
        height: auto !important;
    }
    
    /* Fix tab content scrolling */
    .stTabs [data-baseweb="tab-panel"] {
        overflow-y: visible !important;
        height: auto !important;
    }
    
    /* Ensure all Streamlit elements are scrollable */
    .stContainer {
        overflow-y: visible !important;
        height: auto !important;
    }
    
    
    /* Professional Header */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem 0;
        margin-bottom: 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(30, 60, 114, 0.15);
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 2rem;
        margin: 0 0 0.5rem 0;
        font-weight: 300;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        font-size: 1rem;
        margin: 0;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Professional Sidebar */
    .sidebar .sidebar-content {
        background: #ffffff;
        border-right: 1px solid #e8ecf0;
        box-shadow: 2px 0 8px rgba(0,0,0,0.04);
    }
    
    .sidebar .sidebar-content .block-container {
        padding: 1.5rem 1rem;
    }
    
    /* Medical Grade Cards */
    .info-card {
        background: #ffffff;
        border-left: 4px solid #28a745;
        border-radius: 0 8px 8px 0;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e8f5e8;
    }
    
    .warning-card {
        background: #fffbf0;
        border-left: 4px solid #ff9800;
        border-radius: 0 8px 8px 0;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #fff3cd;
    }
    
    .error-card {
        background: #fff5f5;
        border-left: 4px solid #dc3545;
        border-radius: 0 8px 8px 0;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #f8d7da;
    }
    
    /* Upload Section */
    .upload-area {
        background: #ffffff;
        padding: 3rem 2rem;
        border-radius: 12px;
        text-align: center;
        border: 2px dashed #cbd5e0;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        min-height: 200px;
        overflow-y: visible;
    }
    
    .upload-area:hover {
        border-color: #1e3c72;
        background: #f8f9ff;
    }
    
    .upload-area h3 {
        font-size: 1.6rem;
        margin-bottom: 1rem;
        font-weight: 400;
        color: #2d3748;
    }
    
    .upload-area p {
        font-size: 1rem;
        color: #718096;
        margin-bottom: 0;
    }
    
    /* Professional Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(30, 60, 114, 0.2);
        text-transform: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(30, 60, 114, 0.3);
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
    }
    
    /* Chat Interface */
    .chat-container {
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid #e8ecf0;
        min-height: 400px;
        overflow-y: visible !important;
        height: auto !important;
    }
    
    /* Chat Messages Container */
    .chat-messages-container {
        min-height: 300px;
        overflow-y: visible !important;
        padding-right: 10px;
        height: auto !important;
    }
    
    /* Force chat messages to be scrollable */
    .stChatMessage {
        overflow-y: visible !important;
        height: auto !important;
    }
    
    .chat-message {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.25rem;
        line-height: 1.6;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    .user-message {
        background: #f7fafc;
        margin-left: 15%;
        border-left: 4px solid #1e3c72;
        border: 1px solid #e2e8f0;
    }
    
    .assistant-message {
        background: #ffffff;
        margin-right: 15%;
        border-left: 4px solid #28a745;
        border: 1px solid #e8f5e8;
    }
    
    /* Form Elements */
    .stSelectbox > div > div {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #1e3c72;
        box-shadow: 0 0 0 3px rgba(30, 60, 114, 0.1);
    }
    
    .stTextInput > div > div > input {
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1e3c72;
        box-shadow: 0 0 0 3px rgba(30, 60, 114, 0.1);
    }
    
    .stTextArea > div > div > textarea {
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #1e3c72;
        box-shadow: 0 0 0 3px rgba(30, 60, 114, 0.1);
    }
    
    /* File Uploader */
    .stFileUploader > div {
        border: 2px dashed #cbd5e0;
        border-radius: 12px;
        background: #f8f9ff;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #1e3c72;
        background: #f0f4ff;
    }
    
    /* Chat Input */
    .stChatInput {
        position: relative;
        background: #ffffff;
        border-top: 1px solid #e8ecf0;
        padding: 1.5rem 0;
        z-index: 100;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.05);
        margin-top: 2rem;
    }
    
    .stChatInput > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f8f9fa;
        padding: 4px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 1.5rem;
        background: transparent;
        border-radius: 6px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1e3c72;
        color: white;
        box-shadow: 0 4px 12px rgba(30, 60, 114, 0.2);
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: #e2e8f0;
    }
    
    /* Metrics */
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e8ecf0;
        text-align: center;
        margin: 0.75rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    /* Status Messages */
    .stSuccess {
        background: #f0fff4;
        border: 1px solid #9ae6b4;
        border-radius: 8px;
        color: #22543d;
    }
    
    .stError {
        background: #fff5f5;
        border: 1px solid #feb2b2;
        border-radius: 8px;
        color: #742a2a;
    }
    
    .stWarning {
        background: #fffbf0;
        border: 1px solid #fbd38d;
        border-radius: 8px;
        color: #744210;
    }
    
    .stInfo {
        background: #ebf8ff;
        border: 1px solid #90cdf4;
        border-radius: 8px;
        color: #2a4365;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e0;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a0aec0;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Professional Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 400;
        color: #2d3748;
    }
    
    p, div, span {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Additional scrolling overrides */
    .stApp > div > div {
        height: auto !important;
        overflow-y: visible !important;
    }
    
    /* Force scrollable content */
    .main .block-container > div > div {
        height: auto !important;
        overflow-y: visible !important;
    }
    
    /* Override all possible height constraints */
    div[data-testid] {
        height: auto !important;
        overflow-y: visible !important;
        max-height: none !important;
    }
    
    /* Force page to be scrollable */
    .stApp > div > div > div {
        height: auto !important;
        overflow-y: visible !important;
        max-height: none !important;
    }
    
    /* Override Streamlit's internal height management */
    .stApp > div > div > div > div {
        height: auto !important;
        overflow-y: visible !important;
        max-height: none !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .user-message, .assistant-message {
            margin-left: 5%;
            margin-right: 5%;
        }
    }
</style>

<script>
// Force scrolling to work properly
document.addEventListener('DOMContentLoaded', function() {
    // Remove height constraints from all containers
    const containers = document.querySelectorAll('.main, .stApp, .block-container, .stContainer');
    containers.forEach(container => {
        container.style.height = 'auto';
        container.style.overflowY = 'visible';
        container.style.maxHeight = 'none';
    });
    
    // Ensure body and html can scroll
    document.body.style.height = 'auto';
    document.body.style.overflowY = 'auto';
    document.documentElement.style.height = 'auto';
    document.documentElement.style.overflowY = 'auto';
    
    // Force Streamlit containers to be scrollable
    const streamlitContainers = document.querySelectorAll('[data-testid]');
    streamlitContainers.forEach(container => {
        container.style.height = 'auto';
        container.style.overflowY = 'visible';
    });
});

// Reapply fixes when Streamlit reruns
window.addEventListener('load', function() {
    setTimeout(() => {
        const containers = document.querySelectorAll('.main, .stApp, .block-container, .stContainer');
        containers.forEach(container => {
            container.style.height = 'auto';
            container.style.overflowY = 'visible';
            container.style.maxHeight = 'none';
        });
    }, 1000);
});
</script>
""", unsafe_allow_html=True)

class ImprovedChatbotApp:
    def __init__(self):
        self.config = Config()
        # API key management is now handled directly in config
        self.initialize_session_state()
        self.initialize_components()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'rag_processor' not in st.session_state:
            st.session_state.rag_processor = None
        if 'documents_loaded' not in st.session_state:
            st.session_state.documents_loaded = False
        if 'response_mode' not in st.session_state:
            st.session_state.response_mode = "Detailed"
        if 'selected_provider' not in st.session_state:
            st.session_state.selected_provider = "Groq"
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = "groq/compound"
    
    def initialize_components(self):
        """Initialize core components"""
        try:
            # API keys are loaded from environment variables via config.py
            # No hardcoded keys for security
            
            self.llm_manager = LLMManager(self.config)
            
            # Try to initialize database, but don't fail if it's not available
            try:
                
                @retry_on_failure(max_retries=2, delay=1.0)
                def init_database():
                    return DatabaseManager()
                
                self.db = init_database()
                self.medical_rag = SimpleMedicalRAG(self.db)
                self.comprehensive_rag = ComprehensiveRAG(self.db)
                st.session_state.db_available = True
                
                # Load conversation history
                try:
                    history = self.db.get_conversation_history(st.session_state.conversation_id)
                    if history:
                        st.session_state.messages = history
                except Exception as history_error:
                    ErrorHandler.log_error(history_error, "load_conversation_history")
                    logger.warning(f"Could not load conversation history: {str(history_error)}")
                    
            except Exception as db_error:
                from utils.error_handler import ErrorHandler
                error_id = ErrorHandler.log_error(db_error, "database_initialization", {
                    'conversation_id': st.session_state.conversation_id
                })
                st.warning(f"Database not available (Error ID: {error_id}): {str(db_error)}")
                st.info("Running in offline mode - data will not be persisted")
                self.db = None
                self.medical_rag = None
                self.comprehensive_rag = None
                st.session_state.db_available = False
            
            self.medical_extractor = MedicalReportExtractor(self.llm_manager)
            
            # Set advanced components to None for now
            self.embedding_manager = None
            self.web_search = None
            self.response_processor = None
            
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            st.stop()
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        with st.sidebar:
            st.markdown("")
            
            # Model selection
            st.markdown("### AI Model")
            available_models = []
            if self.config.is_configured('Groq'):
                available_models.append("Groq (Compound)")

            
            if available_models:
                selected_model = st.selectbox(
                    "Choose AI model:",
                    available_models,
                    index=0
                )
                # Map display names to internal names
                model_mapping = {
                    "Groq (Compound)": ("Groq", "groq/compound"),

                }
                st.session_state.selected_provider, st.session_state.selected_model = model_mapping[selected_model]
            else:
                st.warning("No AI models configured. Please add API keys.")
                st.session_state.selected_provider = "Groq"
                st.session_state.selected_model = "groq/compound"
            
            # Response mode selector
            st.markdown("### Response Mode")
            response_mode = st.selectbox(
                "Choose response style:",
                ["Concise", "Detailed", "Technical"],
                index=["Concise", "Detailed", "Technical"].index(st.session_state.response_mode)
            )
            st.session_state.response_mode = response_mode
            
            # Help section
            st.markdown("### ❓ Help")
            with st.expander("How to use"):
                st.markdown("""
                1. **Upload Documents**: Upload PDF, TXT, DOCX, or MD files
                2. **Process Documents**: Click "Process Documents" to extract medical data
                3. **Ask Questions**: Use the chat interface to ask about medical data
                4. **Response Modes**: 
                   - **Concise**: Brief, to-the-point answers
                   - **Detailed**: Comprehensive explanations
                   - **Technical**: Medical terminology and detailed analysis
                """)
    
    def render_header(self):
        """Render the application header"""
        st.markdown("""
        <div class="main-header">
            <h1>Medical AI Assistant</h1>
            <p>Advanced Medical Document Analysis with AI-Powered Clinical Insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_document_upload(self):
        """Render the document upload section"""
        st.markdown("## Document Upload & Processing")
        
        st.markdown("""
        <div class="upload-area">
            <h3>Upload Medical Documents</h3>
            <p>Support for PDF, TXT, DOCX, and MD files. Advanced medical data extraction and analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'txt', 'docx', 'md'],
            accept_multiple_files=True,
            key="doc_upload"
        )
        
        if uploaded_files and not st.session_state.documents_loaded:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Process Documents", type="primary", use_container_width=True):
                    self.process_documents(uploaded_files)
    
    def process_documents(self, uploaded_files):
        """Process uploaded documents with size limits and error handling"""
        with st.spinner("Processing documents..."):
            try:
                processed_count = 0
                medical_count = 0
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Get configuration limits
                max_file_size = self.config.UPLOAD_CONFIG.get('max_file_size', 5 * 1024 * 1024)
                max_content_length = self.config.UPLOAD_CONFIG.get('max_content_length', 1000000)
                
                for i, file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing {file.name}...")
                        
                        # Check file size
                        if file.size > max_file_size:
                            st.error(f"File {file.name} is too large ({file.size} bytes). Maximum allowed: {max_file_size} bytes")
                            continue
                        
                        # Read file content
                        from models.embeddings import process_uploaded_file, chunk_text
                        
                        # Debug: Show file info
                        st.info(f"Processing file: {file.name} (Size: {file.size} bytes)")
                        
                        content, filename = process_uploaded_file(file)
                        
                        # Check content length
                        if len(content) > max_content_length:
                            st.warning(f"Content too large ({len(content)} chars), truncating to {max_content_length} chars")
                            content = content[:max_content_length] + "..."
                        
                        # Debug: Show extracted content length
                        st.info(f"Extracted content length: {len(content)} characters")
                        if len(content) > 0:
                            st.text_area(f"First 500 characters of {filename}:", content[:500], height=100)
                        
                        if not content.strip():
                            st.warning(f"No content extracted from {filename}")
                            continue
                        
                        # Chunk the content with smaller chunks
                        chunk_size = self.config.RAG_CONFIG.get('chunk_size', 500)
                        chunk_overlap = self.config.RAG_CONFIG.get('chunk_overlap', 100)
                        chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        
                        # Store document in database (if available)
                        if st.session_state.get('db_available', False):
                            document_id = self.db.store_document(filename, content, chunks)
                            processed_count += 1
                        else:
                            document_id = None
                            processed_count += 1
                            st.info("Document processed (not stored - database unavailable)")
                        
                        # Try to extract medical data
                        try:
                            status_text.text(f"Analyzing {filename} for medical data...")
                            extracted_data = self.medical_extractor.extract_from_text(content)
                            
                            if extracted_data and extracted_data.get('patient', {}).get('name'):
                                # Store medical data (if database available)
                                if st.session_state.get('db_available', False) and document_id:
                                    self.db.store_medical_report(document_id, extracted_data)
                                    medical_count += 1
                                else:
                                    medical_count += 1
                                    st.info("Medical data extracted (not stored - database unavailable)")
                                
                                patient_name = extracted_data['patient']['name']
                                
                                # Show success with enhanced styling
                                st.markdown(f"""
                                <div class="info-card">
                                    <h4>Medical Data Extracted Successfully!</h4>
                                    <p><strong>Patient:</strong> {patient_name}</p>
                                    <p><strong>Age:</strong> {extracted_data['patient'].get('age', 'N/A')}</p>
                                    <p><strong>Sex:</strong> {extracted_data['patient'].get('sex', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show test counts
                                test_counts = {}
                                for category in ['biochemistry', 'liverFunction', 'lipidProfile', 'thyroidProfile', 'immunoassay', 'other']:
                                    count = len(extracted_data.get(category, []))
                                    if count > 0:
                                        test_counts[category] = count
                                
                                if test_counts:
                                    st.markdown("**Extracted Test Categories:**")
                                    cols = st.columns(len(test_counts))
                                    for i, (category, count) in enumerate(test_counts.items()):
                                        with cols[i]:
                                            st.metric(category.replace('Function', ' Func'), count)
                            else:
                                st.markdown(f"""
                                <div class="warning-card">
                                    <h4>No Medical Data Found</h4>
                                    <p>Could not extract structured medical data from {filename}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.markdown(f"""
                            <div class="error-card">
                                <h4>Extraction Error</h4>
                                <p>Error extracting medical data from {filename}: {str(e)}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-card">
                            <h4>Processing Error</h4>
                            <p>Error processing {file.name}: {str(e)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Update progress
                    if len(uploaded_files) > 0:
                        progress_value = i / len(uploaded_files)
                        # Ensure progress value is between 0 and 1
                        progress_value = max(0.0, min(1.0, progress_value))
                        logger.info(f"Progress update: {i}/{len(uploaded_files)} = {progress_value}")
                        progress_bar.progress(progress_value)
                    else:
                        progress_bar.progress(1.0)
                
                st.session_state.documents_loaded = True
                
                # Final success message
                st.markdown(f"""
                <div class="info-card">
                    <h4>Processing Complete!</h4>
                    <p><strong>Documents Processed:</strong> {processed_count}</p>
                    <p><strong>Medical Reports Extracted:</strong> {medical_count}</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-card">
                    <h4>Processing Failed</h4>
                    <p>Error processing documents: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def render_chat_messages(self):
        """Render the chat messages (without input)"""
        st.markdown("## AI Medical Assistant")
        
        # Create scrollable container for chat messages
        st.markdown('<div class="chat-messages-container">', unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("Sources"):
                        for source in message["sources"]:
                            st.markdown(f"• {source}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input will be handled outside tabs

    def generate_response(self, prompt):
        """Generate response using available tools and LLM"""
        from utils.error_handler import handle_errors, ErrorHandler
        
        @handle_errors(context="generate_response", default_return="I apologize, but I encountered an error while processing your request. Please try again.")
        def _generate_response_internal(prompt):
            return self._generate_response_core(prompt)
        
        return _generate_response_internal(prompt)
    
    def _generate_response_core(self, prompt):
        """Core response generation logic with comprehensive error handling"""
        try:
            # Search all data using comprehensive RAG (if available)
            if st.session_state.get('db_available', False) and self.comprehensive_rag:
                try:
                    # Use comprehensive RAG to search all tables
                    search_results = self.comprehensive_rag.search_all_data(prompt)
                    
                    # Show user what data was found
                    if search_results['contexts']:
                        st.info(f"Found {search_results['total_results']} relevant records, optimized to {search_results['optimized_results']} for analysis")
                    
                    # Create enhanced prompt with all available data
                    enhanced_prompt = self._create_comprehensive_prompt(prompt, search_results)
                    
                    # Pass the medical data as context to the LLM
                    medical_context = search_results.get('contexts', [])
                    
                    # For comprehensive test analysis, create a structured summary
                    if any(keyword in prompt.lower() for keyword in ['all tests', 'comprehensive', 'complete analysis', 'every test']):
                        # Create comprehensive test summary covering all test types
                        comprehensive_summary = self.comprehensive_rag.create_comprehensive_test_summary(prompt)
                        if comprehensive_summary:
                            medical_context = [comprehensive_summary]
                    
                    # Generate response using selected model
                    response = self.llm_manager.generate_response(
                        prompt=prompt,  # Use original prompt, not enhanced
                        context=medical_context,  # Pass medical data as context
                        response_mode=st.session_state.response_mode,
                        provider=st.session_state.selected_provider,
                        model=st.session_state.selected_model
                    )
                except Exception as rag_error:
                    logger.error(f"RAG search failed: {str(rag_error)}")
                    # Fallback to basic response
                    response = self.llm_manager.generate_response(
                        prompt=prompt,
                        context=[],  # No context available in fallback
                        response_mode=st.session_state.response_mode,
                        provider=st.session_state.selected_provider,
                        model=st.session_state.selected_model
                    )
                
                # If response is too short or generic, enhance with web search
                if len(response) < 100 or "I don't have" in response.lower() or "no information" in response.lower():
                    try:
                        # Use Serper to get additional information
                        web_search = WebSearchTool(self.config)
                        web_search_results = web_search.search(prompt, max_results=3)
                        
                        if web_search_results and web_search_results.get('contexts'):
                            web_context = web_search_results['contexts']
                            # Combine medical context with web context
                            combined_context = medical_context + web_context
                            
                            response = self.llm_manager.generate_response(
                                prompt=prompt,
                                context=combined_context,
                                response_mode=st.session_state.response_mode,
                                provider=st.session_state.selected_provider,
                                model=st.session_state.selected_model
                            )
                    except Exception as e:
                        from utils.error_handler import ErrorHandler
                        error_id = ErrorHandler.log_error(e, "web_search_enhancement", {
                            'prompt': prompt[:100],
                            'response_length': len(response)
                        })
                        logger.error(f"Error in web search enhancement (ID: {error_id}): {str(e)}")
                
                return {
                    "content": response,
                    "sources": search_results.get('sources', [])
                }
            else:
                # Fallback: Generate response without RAG
                enhanced_prompt = f"""You are an intelligent medical assistant AI. 
Provide helpful, accurate, and empathetic responses about medical information.

RESPONSE MODE: {st.session_state.response_mode}

IMPORTANT GUIDELINES:
- Always remind users that this is for informational purposes only and not a substitute for professional medical advice
- If abnormal values are found, suggest consulting with a healthcare provider
- Explain medical terms in simple language (unless in Technical mode)
- Be encouraging and supportive in your tone
- Note: I don't have access to specific medical records at the moment

User Question: {prompt}

Please provide a {st.session_state.response_mode.lower()} response. If you identify any medical concerns, recommend consulting a healthcare professional."""
                
                response = self.llm_manager.generate_response(
                    prompt=enhanced_prompt,
                    context=[],
                    response_mode=st.session_state.response_mode,
                    provider=st.session_state.selected_provider,
                    model=st.session_state.selected_model
                )
                
                return {
                    "content": response,
                    "sources": []
                }
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            logger.exception("Error in generate_response")
            return {
                "content": "I apologize, but I encountered an error while processing your request. Please try again.",
                "sources": []
            }
    
    def create_enhanced_prompt(self, user_query, contexts):
        """Create an enhanced prompt based on response mode"""
        context_str = "\n".join(contexts) if contexts else "No specific medical data found."
        
        mode_instructions = {
            "Concise": "Provide a brief, to-the-point answer. Focus on key findings and recommendations.",
            "Detailed": "Provide a comprehensive explanation with context and detailed analysis.",
            "Technical": "Use medical terminology and provide detailed technical analysis with specific values and ranges."
        }
        
        mode_instruction = mode_instructions.get(st.session_state.response_mode, mode_instructions["Detailed"])
        
        return f"""You are an intelligent medical assistant AI. You have access to medical lab reports and patient data. 
Provide helpful, accurate, and empathetic responses about medical information.

RESPONSE MODE: {st.session_state.response_mode}
{mode_instruction}

IMPORTANT GUIDELINES:
- Always remind users that this is for informational purposes only and not a substitute for professional medical advice
- If abnormal values are found, suggest consulting with a healthcare provider
- Explain medical terms in simple language (unless in Technical mode)
- Be encouraging and supportive in your tone
- Use the provided medical data to give specific, relevant answers

Available Medical Context:
{context_str}

User Question: {user_query}

Please provide a {st.session_state.response_mode.lower()} response based on the available medical data. If you identify any abnormal values, explain what they might indicate and recommend consulting a healthcare professional."""
    
    def _create_comprehensive_prompt(self, user_prompt: str, search_results: dict) -> str:
        """Create comprehensive prompt with data from all database tables"""
        try:
            # Get database statistics
            db_stats = self.comprehensive_rag.get_database_stats()
            
            # Build context from all sources
            context_parts = []
            
            if search_results.get('contexts'):
                context_parts.append("RELEVANT DATA FROM DATABASE:")
                context_parts.append("=" * 50)
                for i, context in enumerate(search_results['contexts'], 1):
                    context_parts.append(f"\n[Source {i}]:")
                    context_parts.append(context)
            
            # Add database statistics
            if db_stats:
                context_parts.append("\n\nDATABASE OVERVIEW:")
                context_parts.append("=" * 30)
                context_parts.append(f"Medical Reports: {db_stats.get('medical_reports', 0)}")
                context_parts.append(f"Medical Tests: {db_stats.get('medical_tests', 0)}")
                context_parts.append(f"Documents: {db_stats.get('documents', 0)}")
                context_parts.append(f"Chat Messages: {db_stats.get('chat_messages', 0)}")
                context_parts.append(f"Users: {db_stats.get('users', 0)}")
                context_parts.append(f"Jobs: {db_stats.get('jobs', 0)}")
                
                recent = db_stats.get('recent_activity', {})
                if recent:
                    context_parts.append(f"\nRecent Activity (7 days):")
                    context_parts.append(f"- New Reports: {recent.get('reports_7_days', 0)}")
                    context_parts.append(f"- New Messages: {recent.get('messages_7_days', 0)}")
                    context_parts.append(f"- New Documents: {recent.get('documents_7_days', 0)}")
            
            # Combine all context (truncate if too long)
            full_context = "\n".join(context_parts) if context_parts else "No relevant data found in database."
            
            # Truncate context if too long
            max_context_length = 2000  # Reduced to prevent "Request Entity Too Large"
            if len(full_context) > max_context_length:
                full_context = full_context[:max_context_length] + "\n... [Context truncated]"
            
            # Create the comprehensive prompt
            comprehensive_prompt = f"""You are an intelligent healthcare assistant AI with access to comprehensive medical data and web search capabilities.

DATABASE CONTEXT:
{full_context}

USER QUERY: {user_prompt}

INSTRUCTIONS:
- Be interactive and engaging - ask follow-up questions when appropriate
- Use specific data from the context to provide personalized insights
- Include specific test results, reference ranges, and abnormal flags when available
- Reference sources when providing information
- Be professional, empathetic, and conversational
- Response mode: {st.session_state.response_mode}
- If you need more information, suggest what additional data would be helpful

Please provide an intelligent, interactive response:"""

            return comprehensive_prompt
            
        except Exception as e:
            logger.error(f"Error creating comprehensive prompt: {str(e)}")
            return f"You are a healthcare assistant. Please help with: {user_prompt}"
    
    def render_chat_input(self):
        """Render the chat input (outside tabs)"""
        # Chat input
        if prompt := st.chat_input("Ask me about the medical documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Store user message in database (if available)
            if st.session_state.get('db_available', False) and self.db:
                try:
                    self.db.store_chat_message("user", prompt, st.session_state.conversation_id)
                except Exception as e:
                    logger.warning(f"Failed to store user message: {str(e)}")
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing documents and generating response..."):
                    response_data = self.generate_response(prompt)
                    
                    st.markdown(response_data["content"])
                    
                    if response_data.get("sources"):
                        with st.expander("Sources"):
                            for source in response_data["sources"]:
                                st.markdown(f"• {source}")
            
                    # Store assistant response (if database available)
                    if st.session_state.get('db_available', False) and self.db:
                        try:
                            self.db.store_chat_message(
                                "assistant",
                                response_data["content"],
                                st.session_state.conversation_id,
                                response_data.get("sources", [])
                            )
                        except Exception as e:
                            logger.warning(f"Failed to store assistant message: {str(e)}")
            
            # Add assistant response to session state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_data["content"],
                "sources": response_data.get("sources", [])
            })
    
    def run(self):
        """Run the main application"""
        try:
            self.render_sidebar()
            self.render_header()
            
            # Create tabs to separate upload and chat
            tab1, tab2 = st.tabs(["Document Upload", "Chat Interface"])
            
            with tab1:
                self.render_document_upload()
            
            with tab2:
                self.render_chat_messages()
                # Add some spacing before chat input
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Chat input outside tabs (required by Streamlit) with proper spacing
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")  # Separator line
            self.render_chat_input()
                
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            st.error(traceback.format_exc())

# Main execution
if __name__ == "__main__":
    app = ImprovedChatbotApp()
    app.run()
