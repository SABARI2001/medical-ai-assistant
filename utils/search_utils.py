import requests
import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import time
import re
from urllib.parse import quote_plus
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseSearchEngine:
    """Base class for search engines"""
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for query and return results"""
        raise NotImplementedError

class SerperSearch(BaseSearchEngine):
    """Serper.dev search engine implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"
        self.headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using Serper API with user-provided code"""
        try:
            # Use the provided Serper code structure
            payload = json.dumps({
                "q": query,
                "location": "Coimbatore, Tamil Nadu, India",
                "gl": "in"
            })
            
            response = requests.request("POST", self.base_url, headers=self.headers, data=payload)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Process organic results
            for item in data.get('organic', [])[:max_results]:
                result = {
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'Serper',
                    'date': item.get('date', ''),
                    'position': item.get('position', 0)
                }
                results.append(result)
            
            logger.info(f"Serper search for '{query}' returned {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Serper search error: {str(e)}")
            return []

class BingSearch(BaseSearchEngine):
    """Bing Search API implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"
        self.headers = {
            'Ocp-Apim-Subscription-Key': api_key
        }
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using Bing API"""
        try:
            params = {
                'q': query,
                'count': max_results,
                'mkt': 'en-US',
                'safesearch': 'Moderate'
            }
            
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Process web pages
            for item in data.get('webPages', {}).get('value', [])[:max_results]:
                result = {
                    'title': item.get('name', ''),
                    'link': item.get('url', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'Bing',
                    'date': item.get('dateLastCrawled', ''),
                    'position': len(results) + 1
                }
                results.append(result)
            
            logger.info(f"Bing search for '{query}' returned {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Bing search error: {str(e)}")
            return []

class DuckDuckGoSearch(BaseSearchEngine):
    """DuckDuckGo search (no API key required)"""
    
    def __init__(self):
        self.base_url = "https://api.duckduckgo.com/"
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo Instant Answer API"""
        try:
            params = {
                'q': query,
                'format': 'json',
                'no_redirect': '1',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(
                self.base_url,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Process related topics
            for item in data.get('RelatedTopics', [])[:max_results]:
                if isinstance(item, dict) and 'Text' in item:
                    result = {
                        'title': item.get('Text', '')[:100] + '...',
                        'link': item.get('FirstURL', ''),
                        'snippet': item.get('Text', ''),
                        'source': 'DuckDuckGo',
                        'date': '',
                        'position': len(results) + 1
                    }
                    results.append(result)
            
            # If no related topics, use abstract
            if not results and data.get('Abstract'):
                result = {
                    'title': data.get('Heading', query),
                    'link': data.get('AbstractURL', ''),
                    'snippet': data.get('Abstract', ''),
                    'source': 'DuckDuckGo',
                    'date': '',
                    'position': 1
                }
                results.append(result)
            
            logger.info(f"DuckDuckGo search for '{query}' returned {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {str(e)}")
            return []

class WebSearchTool:
    """Main web search tool that manages multiple search engines"""
    
    def __init__(self, config):
        self.config = config
        self.search_engines = self._initialize_search_engines()
        self.search_cache = {}  # Simple cache for recent searches
        self.cache_ttl = 300  # 5 minutes cache TTL
    
    def _initialize_search_engines(self) -> Dict[str, BaseSearchEngine]:
        """Initialize available search engines"""
        engines = {}
        
        # Serper
        serper_key = self.config.get_api_key('serper')
        if serper_key:
            engines['serper'] = SerperSearch(serper_key)
            logger.info("Initialized Serper search engine")
        
        # Bing
        bing_key = self.config.get_api_key('bing')
        if bing_key:
            engines['bing'] = BingSearch(bing_key)
            logger.info("Initialized Bing search engine")
        
        # DuckDuckGo (always available, no API key needed)
        engines['duckduckgo'] = DuckDuckGoSearch()
        logger.info("Initialized DuckDuckGo search engine")
        
        return engines
    
    def get_available_engines(self) -> List[str]:
        """Get list of available search engines"""
        return list(self.search_engines.keys())
    
    def search(self, query: str, engine: str = None, max_results: int = None) -> Dict[str, Any]:
        """Perform web search using specified or best available engine"""
        try:
            # Use default engine if not specified
            if not engine:
                engine = self._get_best_engine()
            
            if engine not in self.search_engines:
                logger.error(f"Search engine '{engine}' not available")
                return {'contexts': [], 'sources': [], 'results': []}
            
            # Check cache first
            cache_key = f"{query}_{engine}_{max_results}"
            if self._is_cache_valid(cache_key):
                logger.info(f"Returning cached results for: {query}")
                return self.search_cache[cache_key]['data']
            
            max_results = max_results or self.config.WEB_SEARCH_CONFIG.get('max_results', 5)
            
            # Perform search
            search_engine = self.search_engines[engine]
            raw_results = search_engine.search(query, max_results)
            
            if not raw_results:
                logger.warning(f"No results found for query: {query}")
                return {'contexts': [], 'sources': [], 'results': []}
            
            # Process results
            processed_results = self._process_search_results(raw_results, query)
            
            # Cache results
            self.search_cache[cache_key] = {
                'data': processed_results,
                'timestamp': time.time()
            }
            
            return processed_results
        
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return {'contexts': [], 'sources': [], 'results': []}
    
    def _get_best_engine(self) -> str:
        """Get the best available search engine"""
        preferred_order = ['serper', 'bing', 'duckduckgo']
        
        for engine in preferred_order:
            if engine in self.search_engines:
                return engine
        
        # Fallback to first available
        return list(self.search_engines.keys())[0] if self.search_engines else None
    
    def _process_search_results(self, raw_results: List[Dict], query: str, search_type: str = "web") -> Dict[str, Any]:
        """Process raw search results into formatted contexts"""
        contexts = []
        sources = []
        
        for result in raw_results:
            title = result.get('title', '').strip()
            snippet = result.get('snippet', '').strip()
            link = result.get('link', '').strip()
            source_name = result.get('source', 'Web')
            date = result.get('date', '').strip()
            
            if title and snippet:
                # Create context with metadata
                context = f"Title: {title}\n"
                if date:
                    context += f"Date: {date}\n"
                context += f"Content: {snippet}"
                contexts.append(context)
                
                # Create source information
                source_info = f"{title}"
                if date:
                    source_info += f" ({date})"
                source_info += f" - {source_name}"
                if link:
                    source_info += f" [{link}]"
                sources.append(source_info)
        
        return {
            'contexts': contexts,
            'sources': sources,
            'results': raw_results,
            'query': query,
            'search_type': search_type,
            'total_results': len(contexts),
            'search_timestamp': datetime.now().isoformat()
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        if cache_key not in self.search_cache:
            return False
        
        cache_time = self.search_cache[cache_key]['timestamp']
        return (time.time() - cache_time) < self.cache_ttl
    
    def clear_cache(self):
        """Clear search cache"""
        self.search_cache.clear()
        logger.info("Search cache cleared")
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search tool statistics"""
        return {
            'available_engines': list(self.search_engines.keys()),
            'cache_size': len(self.search_cache),
            'best_engine': self._get_best_engine(),
            'cache_ttl_minutes': self.cache_ttl / 60
        }

# Query processing utilities
class QueryProcessor:
    """Helper class for processing and optimizing search queries"""
    
    @staticmethod
    def optimize_query(query: str) -> str:
        """Optimize query for better search results"""
        # Remove common words that don't add search value
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an'}
        
        words = query.lower().split()
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        optimized = ' '.join(filtered_words)
        return optimized if optimized else query
    
    @staticmethod
    def extract_key_terms(query: str) -> List[str]:
        """Extract key terms from query"""
        # Simple implementation - can be enhanced with NLP
        import re
        
        # Find quoted terms
        quoted_terms = re.findall(r'"([^"]*)"', query)
        
        # Find other significant words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'}
        significant_words = [word for word in words if word not in stop_words]
        
        return quoted_terms + significant_words[:5]  # Limit to top 5 terms
    
    @staticmethod
    def suggest_related_queries(query: str) -> List[str]:
        """Suggest related search queries"""
        key_terms = QueryProcessor.extract_key_terms(query)
        
        if not key_terms:
            return []
        
        # Generate variations
        suggestions = []
        main_term = key_terms[0] if key_terms else query
        
        variations = [
            f"{main_term} definition",
            f"{main_term} examples",
            f"how to {main_term}",
            f"{main_term} benefits",
            f"{main_term} vs"
        ]
        
        return variations[:3]

# Streamlit interface helpers
class WebSearchInterface:
    """Streamlit interface helpers for web search"""
    
    @staticmethod
    def display_search_results(results: Dict[str, Any]):
        """Display web search results in Streamlit"""
        if not results.get('contexts'):
            st.info("No web search results found.")
            return
        
        st.success(f"Found {results.get('total_results', 0)} web results")
        
        raw_results = results.get('results', [])
        
        for i, (context, raw_result) in enumerate(zip(results['contexts'], raw_results)):
            with st.expander(f"🌐 Result {i+1}: {raw_result.get('title', 'No title')[:60]}..."):
                st.markdown(f"**Source:** {raw_result.get('source', 'Unknown')}")
                if raw_result.get('link'):
                    st.markdown(f"**Link:** [{raw_result['link']}]({raw_result['link']})")
                if raw_result.get('date'):
                    st.markdown(f"**Date:** {raw_result['date']}")
                
                st.markdown("**Content:**")
                st.write(raw_result.get('snippet', ''))
    
    @staticmethod
    def search_interface(web_search_tool: WebSearchTool) -> Optional[Dict[str, Any]]:
        """Create web search interface"""
        st.subheader("🌐 Web Search")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Enter search query",
                placeholder="e.g., latest developments in artificial intelligence"
            )
        
        with col2:
            engine = st.selectbox(
                "Search Engine",
                web_search_tool.get_available_engines(),
                index=0
            )
        
        if search_query and st.button("Search Web", type="primary"):
            with st.spinner("Searching the web..."):
                results = web_search_tool.search(search_query, engine=engine)
                
                if results.get('contexts'):
                    WebSearchInterface.display_search_results(results)
                    return results
                else:
                    st.error("No results found. Try a different query or search engine.")
        
        return None

# Test function
def test_web_search():
    """Test web search functionality"""
    from config.config import Config
    
    config = Config()
    web_search = WebSearchTool(config)
    
    print("Available search engines:", web_search.get_available_engines())
    print("Search stats:", web_search.get_search_stats())
    
    # Test search
    query = "artificial intelligence latest news"
    print(f"\nSearching for: {query}")
    
    results = web_search.search(query, max_results=3)
    print(f"Found {len(results.get('contexts', []))} results")
    
    for i, context in enumerate(results.get('contexts', [])[:2]):
        print(f"\nResult {i+1}:")
        print(context[:200] + "..." if len(context) > 200 else context)

if __name__ == "__main__":
    test_web_search()