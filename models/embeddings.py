import numpy as np
import openai
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any, Optional, Tuple
import logging
import pickle
import os
from abc import ABC, abstractmethod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseEmbedding(ABC):
    """Base class for embedding models"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        pass

class SentenceTransformerEmbedding(BaseEmbedding):
    """Sentence Transformer embedding implementation"""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        try:
            import torch
            import os
            # Set environment variables to force CPU and disable CUDA
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["USE_TORCH"] = "1"
            
            # Initialize model on CPU
            self.model = SentenceTransformer(model_name)
            self.model.to('cpu')  # Explicitly move to CPU
            self.model_name = model_name
            logger.info(f"Loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model: {str(e)}")
            raise e
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings"""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise e
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()

class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding implementation"""
    
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model_name = model_name
        if self.api_key:
            openai.api_key = self.api_key
        else:
            logger.warning("OpenAI API key not provided")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using OpenAI API"""
        try:
            if not self.api_key:
                raise ValueError("OpenAI API key not configured")
            
            response = openai.embeddings.create(
                input=texts,
                model=self.model_name
            )
            
            embeddings = np.array([item.embedding for item in response.data])
            return embeddings
        
        except Exception as e:
            logger.error(f"Error with OpenAI embeddings: {str(e)}")
            raise e
    
    def get_dimension(self) -> int:
        """Get embedding dimension for the model"""
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimension_map.get(self.model_name, 1536)

class VectorStore:
    """Vector store implementation using FAISS"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        self.texts = []
        self.metadata = []
    
    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict] = None):
        """Add documents to the vector store"""
        try:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            self.index.add(embeddings)
            self.texts.extend(texts)
            
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{"source": f"doc_{i}"} for i in range(len(texts))])
            
            logger.info(f"Added {len(texts)} documents to vector store")
        
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise e
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """Search for similar documents"""
        try:
            # Normalize query embedding
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            
            scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
            
            results = []
            result_scores = []
            result_metadata = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:  # Valid result
                    results.append(self.texts[idx])
                    result_scores.append(float(score))
                    result_metadata.append(self.metadata[idx])
            
            return results, result_scores, result_metadata
        
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise e
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        try:
            data = {
                'index': faiss.serialize_index(self.index),
                'texts': self.texts,
                'metadata': self.metadata,
                'dimension': self.dimension
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Vector store saved to {filepath}")
        
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise e
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.index = faiss.deserialize_index(data['index'])
            self.texts = data['texts']
            self.metadata = data['metadata']
            self.dimension = data['dimension']
            
            logger.info(f"Vector store loaded from {filepath}")
        
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise e

class EmbeddingManager:
    """Manager class for handling embeddings and vector operations"""
    
    def __init__(self, config):
        self.config = config
        self.embedding_model = self._initialize_embedding_model()
        self.vector_store = None
    
    def _initialize_embedding_model(self) -> BaseEmbedding:
        """Initialize embedding model based on configuration"""
        try:
            # Try OpenAI embeddings first if API key is available
            openai_key = self.config.get_api_key('openai')
            if openai_key:
                logger.info("Using OpenAI embeddings")
                return OpenAIEmbedding(
                    api_key=openai_key,
                    model_name=self.config.MODEL_CONFIGS['OpenAI'].get('embedding_model', 'text-embedding-3-small')
                )
            
            # Fall back to SentenceTransformer
            logger.info("Using SentenceTransformer embeddings")
            return SentenceTransformerEmbedding(
                model_name=self.config.RAG_CONFIG.get('embedding_model', 'all-MiniLM-L6-v2')
            )
        
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            # Fallback to basic SentenceTransformer
            return SentenceTransformerEmbedding()
    
    def create_vector_store(self):
        """Create a new vector store"""
        dimension = self.embedding_model.get_dimension()
        self.vector_store = VectorStore(dimension)
        return self.vector_store
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings"""
        return self.embedding_model.encode(texts)
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query"""
        return self.embedding_model.encode([query])
    
    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Add documents to vector store"""
        if not self.vector_store:
            self.create_vector_store()
        
        try:
            embeddings = self.encode_texts(texts)
            self.vector_store.add_documents(texts, embeddings, metadata)
            logger.info(f"Successfully added {len(texts)} documents to vector store")
        
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise e
    
    def search_documents(self, query: str, k: int = None, threshold: float = None) -> Dict[str, Any]:
        """Search for relevant documents"""
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return {"texts": [], "scores": [], "metadata": []}
        
        try:
            k = k or self.config.RAG_CONFIG.get('top_k_results', 5)
            threshold = threshold or self.config.RAG_CONFIG.get('similarity_threshold', 0.7)
            
            query_embedding = self.encode_query(query)
            texts, scores, metadata = self.vector_store.search(query_embedding, k)
            
            # Filter by threshold
            filtered_results = []
            filtered_scores = []
            filtered_metadata = []
            
            for text, score, meta in zip(texts, scores, metadata):
                if score >= threshold:
                    filtered_results.append(text)
                    filtered_scores.append(score)
                    filtered_metadata.append(meta)
            
            return {
                "texts": filtered_results,
                "scores": filtered_scores,
                "metadata": filtered_metadata
            }
        
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return {"texts": [], "scores": [], "metadata": []}
    
    def save_vector_store(self, filepath: str):
        """Save vector store to disk"""
        if self.vector_store:
            self.vector_store.save(filepath)
    
    def load_vector_store(self, filepath: str):
        """Load vector store from disk"""
        if not self.vector_store:
            dimension = self.embedding_model.get_dimension()
            self.vector_store = VectorStore(dimension)
        
        self.vector_store.load(filepath)
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        return {
            "model_type": type(self.embedding_model).__name__,
            "dimension": self.embedding_model.get_dimension(),
            "vector_store_size": len(self.vector_store.texts) if self.vector_store else 0
        }

# Utility functions
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < text_length:
            # Look for sentence endings near the chunk boundary
            for i in range(min(50, text_length - end)):
                if text[end + i] in '.!?':
                    end = end + i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - chunk_overlap
        if start >= text_length:
            break
    
    return chunks

def process_uploaded_file(uploaded_file) -> Tuple[str, str]:
    """Process uploaded file and extract text"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'txt':
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            content = str(uploaded_file.read(), "utf-8")
            return content, uploaded_file.name
        
        elif file_extension == 'pdf':
            import PyPDF2
            from io import BytesIO
            
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
            content = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        content += f"Page {page_num + 1}:\n{page_text}\n\n"
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                    continue
            
            if not content.strip():
                raise ValueError("No text could be extracted from the PDF file")
            
            return content.strip(), uploaded_file.name
        
        elif file_extension in ['docx']:
            from python_docx import Document
            from io import BytesIO
            
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            doc = Document(BytesIO(uploaded_file.read()))
            content = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content += paragraph.text + "\n"
            
            if not content.strip():
                raise ValueError("No text could be extracted from the DOCX file")
            
            return content.strip(), uploaded_file.name
        
        elif file_extension == 'md':
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            content = str(uploaded_file.read(), "utf-8")
            return content, uploaded_file.name
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    except Exception as e:
        logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        raise e

# Test function
def test_embedding_manager():
    """Test function for Embedding Manager"""
    from config.config import Config
    
    config = Config()
    embedding_manager = EmbeddingManager(config)
    
    print("Embedding Info:")
    print(embedding_manager.get_embedding_info())
    
    # Test with sample documents
    test_documents = [
        "This is a test document about machine learning and artificial intelligence.",
        "Python is a popular programming language for data science and web development.",
        "Natural language processing helps computers understand human language."
    ]
    
    print("\nAdding test documents...")
    embedding_manager.add_documents(test_documents)
    
    print("Updated Embedding Info:")
    print(embedding_manager.get_embedding_info())
    
    # Test search
    print("\nTesting search...")
    query = "What is machine learning?"
    results = embedding_manager.search_documents(query)
    
    print(f"Query: {query}")
    print(f"Found {len(results['texts'])} results:")
    for i, (text, score) in enumerate(zip(results['texts'], results['scores'])):
        print(f"{i+1}. Score: {score:.3f} - {text[:100]}...")

if __name__ == "__main__":
    test_embedding_manager()