# NeoStats AI Chatbot - Intelligent Assistant

A powerful, multi-modal AI chatbot built with Streamlit that combines RAG (Retrieval-Augmented Generation), web search, and multiple LLM providers to deliver intelligent, contextual responses.

## 🚀 Features

### Core Capabilities
- **Multi-LLM Support**: OpenAI GPT, Groq (including Compound model), Google Gemini
- **RAG Integration**: Upload and search through your own documents
- **Live Web Search**: Real-time web search with multiple search engines
- **Response Modes**: Switch between Concise and Detailed responses
- **Streaming Responses**: Real-time streaming for Groq Compound model

### Advanced Features
- **Document Processing**: PDF, DOCX, TXT, MD file support
- **Vector Similarity Search**: Intelligent document retrieval using embeddings
- **Multiple Search Engines**: Serper, Bing, DuckDuckGo support
- **Response Enhancement**: Automatic formatting and source citation
- **Smart Caching**: Efficient caching for documents and search results

## 🏗️ Architecture

```
project/
├── config/
│   └── config.py           # Configuration management
├── models/
│   ├── llm.py             # LLM providers (OpenAI, Groq, Gemini)
│   └── embeddings.py      # Embedding models and vector store
├── utils/
│   ├── rag_utils.py       # RAG processing utilities
│   ├── search_utils.py    # Web search functionality
│   └── response_processor.py  # Response enhancement
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd neostats-ai-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

## 🔑 API Keys Setup

The application supports multiple AI providers and search engines. Add your API keys to the `.env` file:

```env
# LLM Providers
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=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
GOOGLE_API_KEY=your_google_key_here

# Search Engines (optional)
SERPER_API_KEY=your_serper_key_here
BING_SEARCH_KEY=your_bing_key_here
```

### Getting API Keys

- **Groq**: Visit [Groq Console](https://console.groq.com/) (Free tier available)
- **OpenAI**: Visit [OpenAI API](https://platform.openai.com/api-keys)
- **Google Gemini**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Serper**: Visit [Serper.dev](https://serper.dev/) (Free tier: 2500 searches)
- **Bing**: Visit [Microsoft Azure Cognitive Services](https://portal.azure.com/)

## 🚀 Quick Start

1. **Run the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the app**
   Open your browser to `http://localhost:8501`

3. **Configure your preferences**
   - Select your preferred LLM provider
   - Choose response mode (Concise/Detailed)
   - Enable RAG and/or Web Search features

4. **Upload documents (optional)**
   - Upload PDF, DOCX, TXT, or MD files
   - Click "Process Documents" to enable RAG search

5. **Start chatting!**
   - Ask questions and get intelligent responses
   - The bot will search your documents and the web as needed

## 🎯 Use Cases

### Business Intelligence
- Upload company documents and get insights
- Research competitors and market trends
- Analyze reports and extract key information

### Research Assistant
- Upload research papers and academic documents
- Get summaries and explanations of complex topics
- Find related information from web sources

### Personal Knowledge Base
- Upload personal documents, notes, manuals
- Quick question-answering from your files
- Enhanced search across your document collection

### Customer Support
- Upload product manuals and documentation
- Provide contextual customer support
- Access to real-time information via web search

## 🔧 Configuration

### Model Selection
- **Groq Compound**: Best for real-time responses with built-in web search
- **OpenAI GPT-4**: Best for complex reasoning and analysis
- **Google Gemini**: Good balance of speed and capability

### Response Modes
- **Concise**: Quick, summarized answers (2-3 sentences)
- **Detailed**: Comprehensive responses with explanations

### Search Settings
- **RAG Only**: Search only uploaded documents
- **Web Search Only**: Search only the internet
- **Combined**: Use both document and web search

## 🚀 Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Add environment variables in settings
5. Deploy!

### Local Deployment
```bash
# Using gunicorn (production)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# Using Docker
docker build -t neostats-chatbot .
docker run -p 8501:8501 neostats-chatbot
```

## 📊 Features Deep Dive

### RAG (Retrieval-Augmented Generation)
- **Document Chunking**: Intelligent text splitting with overlap
- **Vector Embeddings**: Semantic search using sentence transformers
- **Similarity Matching**: Find most relevant document sections
- **Multiple File Types**: PDF, DOCX, TXT, MD support

### Web Search Integration
- **Multiple Engines**: Serper, Bing, DuckDuckGo
- **Query Optimization**: