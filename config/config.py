import os
from dataclasses import dataclass
from typing import Dict, Any, List
from dotenv import load_dotenv

@dataclass
class Config:
    """Configuration class for the chatbot application"""
    
    def __init__(self):
        # Load environment variables from .env file (if it exists)
        load_dotenv('.env')
        
        # API Keys - Use environment variables or set them here
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        self.GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
        self.SERPER_API_KEY = os.getenv('SERPER_API_KEY', '')  # For web search
        self.BING_SEARCH_KEY = os.getenv('BING_SEARCH_KEY', '')  # Alternative web search
        
        # Model configurations
        self.MODEL_CONFIGS = {
            'OpenAI': {
                'models': ['gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo'],
                'embedding_model': 'text-embedding-3-small',
                'max_tokens': 4000,
                'temperature': 0.7
            },
            'Groq': {
                'models': ['groq/compound'],
                'max_tokens': 4100,
                'temperature': 1.0,
                'compound_tools': ['browser_automation', 'web_search']
            },
            'Google Gemini': {
                'models': ['gemini-pro', 'gemini-pro-vision'],
                'max_tokens': 4000,
                'temperature': 0.7
            }
        }
        
        # RAG Configuration
        self.RAG_CONFIG = {
            'chunk_size': 600,  # Balanced chunk size for better context
            'chunk_overlap': 150,  # Better overlap for context continuity
            'top_k_results': 5,  # More results for better context
            'similarity_threshold': 0.7,
            'embedding_model': 'all-mpnet-base-v2',
            'max_context_length': 3000,  # Increased context length for better responses
            'max_prompt_length': 2000  # Increased prompt length for better questions
        }
        
        # Web Search Configuration
        self.WEB_SEARCH_CONFIG = {
            'max_results': 5,
            'search_timeout': 10,
            'default_engine': 'serper'  # or 'bing'
        }
        
        # Response Configuration
        self.RESPONSE_CONFIG = {
            'concise_max_length': 150,
            'detailed_max_length': 800,
            'system_prompt_template': """You are an intelligent AI assistant. Based on the context provided, 
            answer the user's question in a {mode} manner. 
            
            Context: {context}
            
            User Question: {question}
            
            Please provide a {mode} response that directly addresses the user's question."""
        }
        
        # File upload settings
        self.UPLOAD_CONFIG = {
            'max_file_size': 5 * 1024 * 1024,  # 5MB (reduced to prevent errors)
            'allowed_extensions': ['.pdf', '.txt', '.docx', '.md', '.csv'],
            'upload_folder': 'uploads/',
            'chunk_processing': True,  # Process large files in chunks
            'max_content_length': 1000000  # 1MB max content length for processing
        }
        
        # Logging configuration
        self.LOGGING_CONFIG = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_file': 'logs/chatbot.log'
        }
    
    def get_api_key(self, provider: str) -> str:
        """Get API key for specific provider"""
        key_mapping = {
            'openai': self.OPENAI_API_KEY,
            'groq': self.GROQ_API_KEY,
            'google': self.GOOGLE_API_KEY,
            'serper': self.SERPER_API_KEY,
            'bing': self.BING_SEARCH_KEY
        }
        return key_mapping.get(provider.lower(), '')
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration and return status"""
        status = {
            'openai': bool(self.OPENAI_API_KEY),
            'groq': bool(self.GROQ_API_KEY),
            'google': bool(self.GOOGLE_API_KEY),
            'web_search': bool(self.SERPER_API_KEY or self.BING_SEARCH_KEY),
        }
        return status
    
    def get_model_config(self, provider: str) -> Dict[str, Any]:
        """Get model configuration for specific provider"""
        return self.MODEL_CONFIGS.get(provider, {})
    
    def is_configured(self, provider: str) -> bool:
        """Check if a specific provider is configured"""
        provider_map = {
            'Groq': 'groq',
            'OpenAI': 'openai',
            'Google': 'google',
            'Serper': 'serper',
            'Bing': 'bing'
        }
        key_name = provider_map.get(provider, provider.lower())
        return bool(getattr(self, f"{key_name.upper()}_API_KEY", ''))
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        providers = []
        if self.is_configured('Groq'):
            providers.append('Groq')
        if self.is_configured('OpenAI'):
            providers.append('OpenAI')
        if self.is_configured('Google'):
            providers.append('Google')
        return providers
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            'uploads/',
            'logs/',
            'data/',
            'temp/'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Environment setup instructions
ENV_TEMPLATE = """
# Copy this to a .env file in your project root
# Add your actual API keys

OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=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
GOOGLE_API_KEY=your_google_api_key_here
SERPER_API_KEY=your_serper_api_key_here
BING_SEARCH_KEY=your_bing_search_key_here
"""

def setup_environment():
    """Setup environment and create necessary files"""
    config = Config()
    config.create_directories()
    
    # Create .env template if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env.template', 'w') as f:
            f.write(ENV_TEMPLATE)
        print("Created .env.template file. Please copy to .env and add your API keys.")
    
    return config

if __name__ == "__main__":
    config = setup_environment()
    print("Configuration setup complete!")
    print("Validation status:", config.validate_config())
