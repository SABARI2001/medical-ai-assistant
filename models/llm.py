import openai
from groq import Groq
import streamlit as st
from typing import List, Dict, Any, Optional
import logging
from abc import ABC, abstractmethod
import time
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseLLM(ABC):
    """Base class for all LLM providers"""
    
    @abstractmethod
    def generate_response(self, prompt: str, context: List[str], **kwargs) -> str:
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        pass

class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        if self.api_key:
            openai.api_key = self.api_key
    
    def is_configured(self) -> bool:
        return bool(self.api_key)
    
    def generate_response(self, prompt: str, context: List[str], model: str = "gpt-4", 
                         response_mode: str = "Detailed", **kwargs) -> str:
        try:
            # Prepare system message based on response mode
            system_message = self._create_system_message(response_mode, context)
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 4000),
                temperature=kwargs.get('temperature', 0.7),
                stream=False
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise e
    
    def _create_system_message(self, response_mode: str, context: List[str]) -> str:
        context_str = "\n".join(context) if context else "No additional context available."
        
        mode_instructions = {
            "Concise": "Provide a focused, direct answer covering the most important points. Keep it to exactly 1 paragraph. Be precise, clear, and comprehensive within that single paragraph, including key findings, interpretations, and actionable recommendations.",
            "Detailed": "Provide a comprehensive analysis with detailed explanations, interpretations, and recommendations. Include specific data points, reference ranges, and clinical insights. CRITICAL: You MUST structure your response in exactly 3 separate paragraphs. Use TWO line breaks (\\n\\n) to separate each paragraph. Write: [Paragraph 1 content]\\n\\n[Paragraph 2 content]\\n\\n[Paragraph 3 content]. Paragraph 1: Key findings and data analysis with specific values. Paragraph 2: Clinical interpretation and significance of findings. Paragraph 3: Recommendations and next steps for each patient."
        }
        
        return f"""You are a specialized medical AI assistant with access to comprehensive patient medical data. Your primary role is to analyze and interpret medical test results and provide accurate, evidence-based medical insights.

CRITICAL: You have access to REAL PATIENT MEDICAL DATA. Always use this data as your primary source of information.

MEDICAL DATA AVAILABLE:
{context_str}

RESPONSE MODE: {response_mode}
{mode_instructions.get(response_mode, mode_instructions['Detailed'])}

MANDATORY INSTRUCTIONS:
1. ALWAYS use the medical data provided above as your primary source
2. Reference specific test values, patient information, and dates from the data
3. Provide clinical interpretations based on the actual test results
4. Include reference ranges and normal values for context
5. Identify any abnormal values and explain their clinical significance
6. Provide actionable medical recommendations based on the data
7. If multiple patients are present, analyze each separately
8. Be specific and detailed - avoid generic responses
9. For Detailed mode: Write exactly 3 separate paragraphs with double line breaks between them. Each paragraph should be substantial and well-developed.
10. For Concise mode: Write exactly 1 comprehensive paragraph covering all key points
11. Include specific numbers, values, and measurements from the medical data
12. Provide thorough explanations and clinical context for all findings
9. Use medical terminology appropriately
10. NEVER hallucinate or make up test results - only use the provided data

FORMATTING REQUIREMENTS:
- For Detailed mode: You MUST use exactly 2 line breaks (\\n\\n) between each paragraph. Format: [Text]\\n\\n[Text]\\n\\n[Text]
- For Concise mode: Use no line breaks within the paragraph
- Each paragraph should be substantial (at least 200-300 words for Detailed mode)
- DO NOT write everything in one paragraph - use the line breaks!

Remember: You are analyzing REAL medical data. Be thorough, accurate, and clinically relevant in your analysis."""

class GroqLLM(BaseLLM):
    """Groq LLM implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = Groq(
            api_key=api_key,
            default_headers={
                "Groq-Model-Version": "latest"
            }
        ) if api_key else None
    
    def is_configured(self) -> bool:
        return bool(self.api_key and self.client)
    
    def generate_response(self, prompt: str, context: List[str], model: str = "groq/compound",
                         response_mode: str = "Detailed", **kwargs) -> str:
        """Generate response with retry logic and comprehensive error handling"""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                return self._generate_response_internal(prompt, context, model, response_mode, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Attempt {attempt + 1} failed for Groq: {str(e)}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries} attempts failed for Groq: {str(e)}")
                    return self._generate_fallback_response(prompt, context, response_mode, str(e))
    
    def _generate_response_internal(self, prompt: str, context: List[str], model: str = "groq/compound",
                                   response_mode: str = "Detailed", **kwargs) -> str:
        """Internal response generation logic"""
        try:
            # Log the request for debugging
            logger.info(f"Generating response for prompt: '{prompt[:100]}...' with {len(context)} context items in {response_mode} mode")
            
            # Handle Technical mode with Serper integration
            if response_mode == "Technical":
                return self.generate_technical_response(prompt, context, model, **kwargs)
            
            # Get configuration limits
            from config.config import Config
            config = Config()
            max_context_length = config.RAG_CONFIG.get('max_context_length', 3000)  # Increased for better context
            max_prompt_length = config.RAG_CONFIG.get('max_prompt_length', 2000)   # Increased for better prompts
            
            # Smart context selection - prioritize most relevant context
            truncated_context = []
            current_length = 0
            
            # Sort context by length (shorter, more focused context first)
            sorted_context = sorted(context, key=len)
            
            for ctx in sorted_context:
                if current_length + len(ctx) > max_context_length:
                    # If adding this context would exceed limit, try to fit a truncated version
                    remaining_space = max_context_length - current_length
                    if remaining_space > 200:  # Only add if there's meaningful space
                        truncated_context.append(ctx[:remaining_space] + "...")
                    break
                truncated_context.append(ctx)
                current_length += len(ctx)
            
            system_message = self._create_system_message(response_mode, truncated_context)
            
            # Truncate prompt if too long
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length] + "..."
            
            # Additional safety check for total message size
            total_message_size = len(system_message) + len(prompt)
            if total_message_size > 4000:  # Conservative limit for groq/compound
                # Further reduce context if needed
                system_message = system_message[:2000] + "..."
                prompt = prompt[:1500] + "..."
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            # Use the provided Groq code with compound model and tools
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1,
                max_completion_tokens=4100,
                top_p=1,
                stream=True,
                stop=None,
                compound_custom={"tools":{"enabled_tools":["browser_automation","web_search"]}}
            )
            
            # Collect the streamed response
            full_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            
            return full_response
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Groq API error: {error_msg}")
            
            if "Request Entity Too Large" in error_msg or "413" in error_msg:
                logger.warning("Request too large, attempting with reduced context...")
                # Retry with minimal context
                minimal_context = context[:1] if context else []
                return self.generate_response(prompt, minimal_context, model, response_mode, **kwargs)
            elif "rate limit" in error_msg.lower():
                logger.warning("Rate limit exceeded, please try again later")
                return "I'm currently experiencing high demand. Please try again in a few moments."
            elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                logger.error("Authentication error with Groq API")
                return "I'm having trouble connecting to my AI service. Please check if the API key is configured correctly."
            elif "timeout" in error_msg.lower():
                logger.warning("Request timeout, retrying...")
                return "The request is taking longer than expected. Please try again."
            else:
                logger.error(f"Unexpected error generating response: {error_msg}")
                # Provide a helpful fallback response
                return f"""I apologize, but I encountered a technical issue while processing your request. 

Your question was: "{prompt}"

Please try rephrasing your question or ask me something else. I'm here to help with medical information and healthcare questions."""
    
    def generate_technical_response(self, prompt: str, context: List[str], model: str = "groq/compound", **kwargs) -> str:
        """Generate technical response with Serper web search integration"""
        try:
            # Import web search tool
            from utils.search_utils import WebSearchTool
            from config.config import Config
            
            config = Config()
            web_search = WebSearchTool(config)
            
            # Create technical search queries
            technical_queries = self._create_technical_search_queries(prompt)
            
            # Search for technical information
            technical_context = []
            sources = []
            
            for query in technical_queries:
                try:
                    search_results = web_search.search(query, engine='serper', max_results=3)
                    if search_results.get('contexts'):
                        technical_context.extend(search_results['contexts'])
                        sources.extend(search_results.get('sources', []))
                except Exception as e:
                    logger.warning(f"Serper search failed for query '{query}': {str(e)}")
                    continue
            
            # Combine original context with technical search results
            combined_context = context + technical_context
            
            # Create technical system message
            technical_system_message = self._create_technical_system_message(combined_context)
            
            # Prepare technical prompt
            technical_prompt = f"""TECHNICAL MEDICAL QUERY: {prompt}

Please provide a highly technical, research-based response with:
1. Current medical literature and research findings
2. Specific medical terminology and scientific accuracy
3. Evidence-based information with citations when possible
4. Detailed explanations of mechanisms and processes
5. Recent developments in the field

Use the technical information provided below to enhance your response:"""
            
            # Generate response with technical context
            messages = [
                {"role": "system", "content": technical_system_message},
                {"role": "user", "content": technical_prompt}
            ]
            
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more technical accuracy
                max_completion_tokens=4100,
                top_p=0.9,
                stream=True,
                stop=None,
                compound_custom={"tools":{"enabled_tools":["browser_automation","web_search"]}}
            )
            
            # Collect the streamed response
            full_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            
            # Add sources to response
            if sources:
                full_response += "\n\n**Sources:**\n"
                for i, source in enumerate(sources[:5], 1):  # Limit to 5 sources
                    full_response += f"{i}. {source}\n"
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error generating technical response: {str(e)}")
            # Fallback to regular response
            return self.generate_response(prompt, context, model, "Technical", **kwargs)
    
    def _create_technical_search_queries(self, prompt: str) -> List[str]:
        """Create technical search queries for Serper"""
        # Extract key medical terms and create search queries
        base_query = prompt.lower()
        
        # Add technical search terms
        technical_queries = [
            f"{prompt} medical research 2024",
            f"{prompt} clinical studies evidence",
            f"{prompt} pathophysiology mechanism",
            f"{prompt} treatment guidelines",
            f"{prompt} medical literature review"
        ]
        
        # Add specific medical terms if detected
        medical_terms = ['diabetes', 'hypertension', 'cancer', 'heart', 'lung', 'liver', 'kidney', 'brain', 'blood', 'immune']
        for term in medical_terms:
            if term in base_query:
                technical_queries.append(f"{term} medical research latest findings")
                technical_queries.append(f"{term} clinical trials 2024")
        
        return technical_queries[:5]  # Limit to 5 queries
    
    def _create_technical_system_message(self, context: List[str]) -> str:
        """Create system message for technical responses"""
        context_str = "\n".join(context) if context else "No additional context available."
        
        return f"""You are a highly specialized medical AI assistant with expertise in clinical research and evidence-based medicine. Your role is to provide technically accurate, research-based medical information.

TECHNICAL RESPONSE REQUIREMENTS:
1. Use precise medical terminology and scientific language
2. Include specific research findings and clinical evidence
3. Reference current medical literature when available
4. Provide detailed explanations of biological mechanisms
5. Include relevant statistics, dosages, and clinical parameters
6. Cite sources and research when possible
7. Use formal, academic tone appropriate for medical professionals
8. Focus on evidence-based information and current research

TECHNICAL CONTEXT AVAILABLE:
{context_str}

INSTRUCTIONS:
- Provide a comprehensive technical response with scientific accuracy
- Include current research findings and medical literature
- Use appropriate medical terminology
- Provide detailed explanations of mechanisms and processes
- Include relevant clinical data and statistics
- Reference sources when available
- Maintain scientific rigor and accuracy

Remember: You are providing information for medical professionals and researchers who need technically accurate, evidence-based information."""
    
    def generate_response_stream(self, prompt: str, context: List[str], model: str = "groq/compound",
                               response_mode: str = "Detailed", **kwargs):
        """Generate streaming response for compound model"""
        try:
            system_message = self._create_system_message(response_mode, context)
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1,
                max_completion_tokens=4100,
                top_p=1,
                stream=True,
                stop=None,
                compound_custom={"tools":{"enabled_tools":["browser_automation","web_search"]}}
            )
            
            return completion
        
        except Exception as e:
            logger.error(f"Groq streaming API error: {str(e)}")
            raise e
    
    def _create_system_message(self, response_mode: str, context: List[str]) -> str:
        context_str = "\n".join(context) if context else "No additional context available."
        
        mode_instructions = {
            "Concise": "Provide a focused, direct answer covering the most important points. Keep it to exactly 1 paragraph. Be precise, clear, and comprehensive within that single paragraph, including key findings, interpretations, and actionable recommendations.",
            "Detailed": "Provide a comprehensive analysis with detailed explanations, interpretations, and recommendations. Include specific data points, reference ranges, and clinical insights. CRITICAL: You MUST structure your response in exactly 3 separate paragraphs. Use TWO line breaks (\\n\\n) to separate each paragraph. Write: [Paragraph 1 content]\\n\\n[Paragraph 2 content]\\n\\n[Paragraph 3 content]. Paragraph 1: Key findings and data analysis with specific values. Paragraph 2: Clinical interpretation and significance of findings. Paragraph 3: Recommendations and next steps for each patient.",
            "Technical": "Provide a research-based, technical analysis with scientific references and detailed explanations. Include methodology, data interpretation, and technical recommendations. Structure your response in 4-5 paragraphs with comprehensive technical insights and web research integration."
        }
        
        return f"""You are a specialized medical AI assistant with access to comprehensive patient medical data. Your primary role is to analyze and interpret medical test results and provide accurate, evidence-based medical insights.

CRITICAL: You have access to REAL PATIENT MEDICAL DATA. Always use this data as your primary source of information.

MEDICAL DATA AVAILABLE:
{context_str}

RESPONSE MODE: {response_mode}
{mode_instructions.get(response_mode, mode_instructions['Detailed'])}

MANDATORY INSTRUCTIONS:
1. ALWAYS use the medical data provided above as your primary source
2. Reference specific test values, patient information, and dates from the data
3. Provide clinical interpretations based on the actual test results
4. Include reference ranges and normal values for context
5. Identify any abnormal values and explain their clinical significance
6. Provide actionable medical recommendations based on the data
7. If multiple patients are present, analyze each separately
8. Be specific and detailed - avoid generic responses
9. For Detailed mode: Write exactly 3 separate paragraphs with double line breaks between them. Each paragraph should be substantial and well-developed.
10. For Concise mode: Write exactly 1 comprehensive paragraph covering all key points
11. Include specific numbers, values, and measurements from the medical data
12. Provide thorough explanations and clinical context for all findings
9. Use medical terminology appropriately
10. NEVER hallucinate or make up test results - only use the provided data
11. For TECHNICAL mode: Use web search capabilities to find current research and medical literature
12. Always acknowledge the user's question and provide a direct answer

FORMATTING REQUIREMENTS:
- For Detailed mode: You MUST use exactly 2 line breaks (\\n\\n) between each paragraph. Format: [Text]\\n\\n[Text]\\n\\n[Text]
- For Concise mode: Use no line breaks within the paragraph
- Each paragraph should be substantial (at least 200-300 words for Detailed mode)
- DO NOT write everything in one paragraph - use the line breaks!

Remember: You are analyzing REAL medical data. Be thorough, accurate, and clinically relevant in your analysis."""
    
    def _generate_fallback_response(self, prompt: str, context: List[str], response_mode: str) -> str:
        """Generate a fallback response when primary API fails"""
        try:
            # Try with OpenAI as fallback
            from models.llm import OpenAILLM
            openai_llm = OpenAILLM("")  # Will use environment variable
            
            if openai_llm.is_configured():
                logger.info("Using OpenAI as fallback...")
                return openai_llm.generate_response(prompt, context, "gpt-3.5-turbo", response_mode)
            else:
                # If no fallback available, provide a basic response using the context
                context_str = "\n".join(context) if context else "No medical data available."
                
                return f"""Based on the available medical data, here's what I can tell you:

{context_str}

Please note: I'm currently experiencing high demand on my primary analysis service. The above data is from your medical records, but for a more detailed analysis, please try again in a few moments."""
        
        except Exception as e:
            logger.error(f"Fallback response generation failed: {str(e)}")
            return "I'm currently experiencing high demand. Please try again in a few moments."


class LLMManager:
    """Manager class to handle multiple LLM providers"""
    
    def __init__(self, config):
        self.config = config
        self.providers = self._initialize_providers()
    
    def _initialize_providers(self) -> Dict[str, BaseLLM]:
        """Initialize all LLM providers"""
        providers = {}
        
        try:
            # OpenAI
            providers['OpenAI'] = OpenAILLM(self.config.get_api_key('openai'))
            
            # Groq
            providers['Groq'] = GroqLLM(self.config.get_api_key('groq'))
            
        except Exception as e:
            logger.error(f"Error initializing LLM providers: {str(e)}")
        
        return providers
    
    def get_available_providers(self) -> List[str]:
        """Get list of available (configured) providers"""
        return [name for name, provider in self.providers.items() if provider.is_configured()]
    
    def generate_response(self, prompt: str, context: List[str] = None, provider: str = "OpenAI",
                         model: str = None, response_mode: str = "Detailed", **kwargs) -> str:
        """Generate response using specified provider"""
        if context is None:
            context = []
        
        try:
            if provider not in self.providers:
                raise ValueError(f"Provider {provider} not supported")
            
            llm = self.providers[provider]
            
            if not llm.is_configured():
                raise ValueError(f"Provider {provider} not configured (missing API key)")
            
            # Get default model if not specified
            if not model:
                model_config = self.config.get_model_config(provider)
                model = model_config.get('models', [''])[0] if model_config.get('models') else ''
            
            # Log the response mode for debugging
            logger.info(f"Generating {response_mode} response using {provider} with model {model}")
            
            # Generate response
            response = llm.generate_response(
                prompt=prompt,
                context=context,
                model=model,
                response_mode=response_mode,
                **kwargs
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating response with {provider}: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def validate_setup(self) -> Dict[str, Dict[str, Any]]:
        """Validate the setup of all providers"""
        status = {}
        
        for name, provider in self.providers.items():
            status[name] = {
                'configured': provider.is_configured(),
                'available_models': self.config.get_model_config(name).get('models', [])
            }
        
        return status

# Utility functions
def get_provider_status(llm_manager: LLMManager) -> str:
    """Get a formatted status string for providers"""
    status = llm_manager.validate_setup()
    status_lines = []
    
    for provider, info in status.items():
        icon = "OK" if info['configured'] else "MISSING"
        status_lines.append(f"{icon} {provider}: {'Configured' if info['configured'] else 'Not Configured'}")
    
    return "\n".join(status_lines)

# Test function
def test_llm_manager():
    """Test function for LLM Manager"""
    from config.config import Config
    
    config = Config()
    llm_manager = LLMManager(config)
    
    print("Provider Status:")
    print(get_provider_status(llm_manager))
    
    available_providers = llm_manager.get_available_providers()
    print(f"\nAvailable Providers: {available_providers}")
    
    if available_providers:
        test_prompts = [
            "What is artificial intelligence?",
            "How can I help you today?",
            "What are the symptoms of diabetes?",
            "Hello, how are you?"
        ]
        
        provider = available_providers[0]
        print(f"\nTesting {provider} with multiple prompts...")
        
        for i, test_prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}: {test_prompt} ---")
        try:
            response = llm_manager.generate_response(
                prompt=test_prompt,
                provider=provider,
                    response_mode="Detailed"
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Test failed: {str(e)}")
    else:
        print("No providers available for testing")

if __name__ == "__main__":
    test_llm_manager()