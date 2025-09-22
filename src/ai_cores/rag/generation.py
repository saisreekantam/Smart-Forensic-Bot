"""
Advanced Response Generation System for Forensic RAG

This module provides sophisticated response generation with multi-LLM support,
forensic-specific prompt engineering, and context-aware answer synthesis.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import re
from abc import ABC, abstractmethod

# LLM integration libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Local imports
from .retrieval import RankedResult, RetrievalQuery

logger = logging.getLogger(__name__)

@dataclass
class GenerationContext:
    """Context for response generation"""
    query: str
    retrieved_results: List[RankedResult]
    conversation_history: Optional[List[Dict[str, str]]] = None
    user_preferences: Optional[Dict[str, Any]] = None
    domain_context: str = "forensic_investigation"
    response_style: str = "professional"  # 'professional', 'detailed', 'concise', 'technical'
    include_sources: bool = True
    include_confidence: bool = True
    max_response_length: int = 1000

@dataclass
class GeneratedResponse:
    """Container for generated responses with metadata"""
    response_text: str
    confidence_score: float
    sources_used: List[str]
    reasoning_chain: Optional[str] = None
    model_used: str = "unknown"
    generation_metadata: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        context: GenerationContext,
        **kwargs
    ) -> GeneratedResponse:
        """Generate response using the LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        pass

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider for response generation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key')
        self.model_name = config.get('model', 'gpt-3.5-turbo')
        self.max_tokens = config.get('max_tokens', 1000)
        self.temperature = config.get('temperature', 0.3)
        
        if self.api_key:
            openai.api_key = self.api_key
    
    def is_available(self) -> bool:
        return OPENAI_AVAILABLE and self.api_key is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "OpenAI",
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
    
    def generate_response(
        self,
        prompt: str,
        context: GenerationContext,
        **kwargs
    ) -> GeneratedResponse:
        """Generate response using OpenAI GPT"""
        
        try:
            # Prepare messages for chat completion
            messages = [
                {
                    "role": "system",
                    "content": self._get_system_prompt(context)
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Add conversation history if available
            if context.conversation_history:
                for msg in context.conversation_history[-5:]:  # Last 5 messages
                    messages.insert(-1, msg)
            
            # Generate response
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                **kwargs
            )
            
            response_text = response.choices[0].message.content
            
            # Extract confidence and sources
            confidence_score = self._estimate_confidence(response, context)
            sources_used = self._extract_sources_from_context(context)
            
            return GeneratedResponse(
                response_text=response_text,
                confidence_score=confidence_score,
                sources_used=sources_used,
                model_used=f"OpenAI-{self.model_name}",
                generation_metadata={
                    "usage": response.usage._asdict() if response.usage else None,
                    "finish_reason": response.choices[0].finish_reason
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return GeneratedResponse(
                response_text=f"Error generating response: {str(e)}",
                confidence_score=0.0,
                sources_used=[],
                model_used=f"OpenAI-{self.model_name}",
                warnings=[f"Generation failed: {str(e)}"]
            )
    
    def _get_system_prompt(self, context: GenerationContext) -> str:
        """Generate system prompt based on context"""
        
        base_prompt = """You are an expert forensic analyst AI assistant specialized in analyzing digital evidence and UFDR (Universal Forensic Data Record) data. Your role is to provide accurate, professional analysis based on retrieved forensic evidence."""
        
        style_prompts = {
            "professional": "Provide clear, professional responses suitable for law enforcement and legal proceedings.",
            "detailed": "Provide comprehensive, detailed analysis with step-by-step reasoning.",
            "concise": "Provide brief, focused responses highlighting key findings.",
            "technical": "Provide technical analysis with detailed forensic terminology and methodologies."
        }
        
        domain_context = """
Context: You are analyzing forensic evidence including:
- Communication records (messages, calls, emails)
- Contact information and relationships
- Digital transactions and cryptocurrency addresses
- Device information and extraction metadata
- Temporal patterns and location data

Guidelines:
1. Base your analysis ONLY on the provided evidence
2. Clearly distinguish between facts and inferences
3. Highlight potential investigative leads
4. Note any limitations or gaps in the evidence
5. Maintain chain of custody considerations
6. Use appropriate forensic terminology
"""
        
        style_addition = style_prompts.get(context.response_style, style_prompts["professional"])
        
        return f"{base_prompt}\n\n{domain_context}\n\n{style_addition}"
    
    def _estimate_confidence(self, response: Any, context: GenerationContext) -> float:
        """Estimate confidence in the generated response"""
        base_confidence = 0.7
        
        # Boost confidence based on number of sources
        source_count = len(context.retrieved_results)
        if source_count >= 5:
            base_confidence += 0.2
        elif source_count >= 3:
            base_confidence += 0.1
        
        # Check for uncertainty phrases in response
        uncertainty_phrases = [
            "uncertain", "unclear", "might", "possibly", "perhaps",
            "appears to", "seems to", "suggest that"
        ]
        
        response_text = response.choices[0].message.content.lower()
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in response_text)
        
        if uncertainty_count > 3:
            base_confidence -= 0.2
        elif uncertainty_count > 1:
            base_confidence -= 0.1
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _extract_sources_from_context(self, context: GenerationContext) -> List[str]:
        """Extract source identifiers from context"""
        sources = []
        for result in context.retrieved_results:
            source = f"{result.result.metadata.data_type}:{result.result.id}"
            if result.result.metadata.source_file:
                source += f" ({result.result.metadata.source_file})"
            sources.append(source)
        return sources

class GoogleProvider(BaseLLMProvider):
    """Google Gemini provider for response generation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key')
        self.model_name = config.get('model', 'gemini-pro')
        self.temperature = config.get('temperature', 0.3)
        self.max_output_tokens = config.get('max_tokens', 1000)
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
    
    def is_available(self) -> bool:
        return GOOGLE_AVAILABLE and self.api_key is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "Google",
            "model": self.model_name,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature
        }
    
    def generate_response(
        self,
        prompt: str,
        context: GenerationContext,
        **kwargs
    ) -> GeneratedResponse:
        """Generate response using Google Gemini"""
        
        try:
            # Prepare full prompt with system context
            system_prompt = self._get_system_prompt(context)
            full_prompt = f"{system_prompt}\n\nUser Query: {prompt}"
            
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                **kwargs
            )
            
            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            response_text = response.text
            
            # Extract confidence and sources
            confidence_score = self._estimate_confidence(response, context)
            sources_used = self._extract_sources_from_context(context)
            
            return GeneratedResponse(
                response_text=response_text,
                confidence_score=confidence_score,
                sources_used=sources_used,
                model_used=f"Google-{self.model_name}",
                generation_metadata={
                    "finish_reason": getattr(response, 'finish_reason', None),
                    "safety_ratings": getattr(response, 'safety_ratings', None)
                }
            )
            
        except Exception as e:
            logger.error(f"Google generation failed: {e}")
            return GeneratedResponse(
                response_text=f"Error generating response: {str(e)}",
                confidence_score=0.0,
                sources_used=[],
                model_used=f"Google-{self.model_name}",
                warnings=[f"Generation failed: {str(e)}"]
            )
    
    def _get_system_prompt(self, context: GenerationContext) -> str:
        """Generate system prompt for Google models"""
        return """You are a forensic analysis AI specialized in digital evidence examination.

Your expertise includes:
- Communication pattern analysis
- Digital transaction tracing
- Entity relationship mapping
- Temporal sequence analysis
- Evidence correlation and verification

Instructions:
1. Analyze only the provided forensic evidence
2. Provide factual, evidence-based responses
3. Clearly separate observations from inferences
4. Highlight investigative opportunities
5. Note evidence limitations and gaps
6. Use professional forensic terminology
7. Maintain objectivity and precision

Format your response clearly with:
- Key Findings
- Analysis Summary  
- Potential Leads
- Evidence Assessment"""
    
    def _estimate_confidence(self, response: Any, context: GenerationContext) -> float:
        """Estimate confidence for Google responses"""
        base_confidence = 0.75
        
        # Check safety ratings and finish reason
        if hasattr(response, 'finish_reason'):
            if response.finish_reason == "STOP":
                base_confidence += 0.1
            elif response.finish_reason in ["SAFETY", "RECITATION"]:
                base_confidence -= 0.2
        
        # Adjust based on retrieved context quality
        avg_similarity = sum(r.result.similarity_score for r in context.retrieved_results) / max(len(context.retrieved_results), 1)
        if avg_similarity > 0.8:
            base_confidence += 0.1
        elif avg_similarity < 0.5:
            base_confidence -= 0.1
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _extract_sources_from_context(self, context: GenerationContext) -> List[str]:
        """Extract source identifiers from context"""
        sources = []
        for result in context.retrieved_results:
            source = f"{result.result.metadata.data_type}:{result.result.id}"
            if result.result.metadata.case_id:
                source += f" (Case: {result.result.metadata.case_id})"
            sources.append(source)
        return sources

class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Transformers provider for local models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model', 'microsoft/DialoGPT-small')
        self.device = config.get('device', 'auto')
        self.max_length = config.get('max_length', 1000)
        self.temperature = config.get('temperature', 0.3)
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        if TRANSFORMERS_AVAILABLE:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize HuggingFace model and tokenizer"""
        try:
            # Determine device
            if self.device == 'auto':
                device = 0 if torch.cuda.is_available() else -1
            else:
                device = self.device
            
            # Initialize pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=device,
                torch_dtype=torch.float16 if device >= 0 else torch.float32
            )
            
            logger.info(f"Initialized HuggingFace model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model: {e}")
            self.pipeline = None
    
    def is_available(self) -> bool:
        return TRANSFORMERS_AVAILABLE and self.pipeline is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "HuggingFace",
            "model": self.model_name,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "device": self.device
        }
    
    def generate_response(
        self,
        prompt: str,
        context: GenerationContext,
        **kwargs
    ) -> GeneratedResponse:
        """Generate response using HuggingFace model"""
        
        if not self.pipeline:
            return GeneratedResponse(
                response_text="HuggingFace model not available",
                confidence_score=0.0,
                sources_used=[],
                model_used=self.model_name,
                warnings=["Model initialization failed"]
            )
        
        try:
            # Prepare prompt
            system_context = "You are a forensic analysis assistant. Analyze the provided evidence professionally."
            full_prompt = f"{system_context}\n\nQuery: {prompt}\n\nResponse:"
            
            # Generate response
            outputs = self.pipeline(
                full_prompt,
                max_length=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            response_text = generated_text[len(full_prompt):].strip()
            
            # Basic confidence estimation
            confidence_score = 0.6  # Lower confidence for local models
            sources_used = self._extract_sources_from_context(context)
            
            return GeneratedResponse(
                response_text=response_text,
                confidence_score=confidence_score,
                sources_used=sources_used,
                model_used=f"HuggingFace-{self.model_name}",
                generation_metadata={
                    "generated_length": len(response_text),
                    "prompt_length": len(full_prompt)
                }
            )
            
        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            return GeneratedResponse(
                response_text=f"Error generating response: {str(e)}",
                confidence_score=0.0,
                sources_used=[],
                model_used=f"HuggingFace-{self.model_name}",
                warnings=[f"Generation failed: {str(e)}"]
            )
    
    def _extract_sources_from_context(self, context: GenerationContext) -> List[str]:
        """Extract source identifiers from context"""
        return [f"{r.result.metadata.data_type}:{r.result.id}" for r in context.retrieved_results]

class ForensicPromptEngine:
    """
    Specialized prompt engineering for forensic analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.templates = self._load_prompt_templates()
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load forensic-specific prompt templates"""
        return {
            "entity_analysis": """
Based on the forensic evidence provided, analyze the following entities and their relationships:

Evidence Context:
{evidence_context}

Query: {query}

Please provide:
1. Entity Identification and Verification
2. Relationship Mapping
3. Investigative Significance
4. Recommended Follow-up Actions

Maintain objectivity and distinguish between confirmed facts and potential inferences.
""",
            
            "timeline_analysis": """
Based on the forensic evidence provided, construct a timeline analysis:

Evidence Context:
{evidence_context}

Query: {query}

Please provide:
1. Chronological Sequence of Events
2. Time Gap Analysis
3. Pattern Identification
4. Temporal Correlations
5. Missing Time Periods

Focus on factual temporal data and note any assumptions made.
""",
            
            "communication_analysis": """
Analyze the communication patterns and content from the forensic evidence:

Evidence Context:
{evidence_context}

Query: {query}

Please provide:
1. Communication Patterns
2. Participant Analysis
3. Content Themes and Significance
4. Technical Indicators
5. Behavioral Insights

Base analysis strictly on provided communication records.
""",
            
            "general_analysis": """
Conduct a comprehensive forensic analysis based on the provided evidence:

Evidence Context:
{evidence_context}

Query: {query}

Please provide:
1. Key Findings Summary
2. Evidence Assessment
3. Investigative Leads
4. Correlations and Patterns
5. Limitations and Gaps

Maintain professional forensic standards and evidence-based conclusions.
"""
        }
    
    def create_prompt(
        self,
        query: str,
        context: GenerationContext,
        query_type: str = "general"
    ) -> str:
        """Create forensic-specific prompt based on query type and context"""
        
        # Select appropriate template
        template_key = f"{query_type}_analysis"
        if template_key not in self.templates:
            template_key = "general_analysis"
        
        template = self.templates[template_key]
        
        # Prepare evidence context
        evidence_context = self._format_evidence_context(context)
        
        # Format prompt
        prompt = template.format(
            evidence_context=evidence_context,
            query=query
        )
        
        return prompt
    
    def _format_evidence_context(self, context: GenerationContext) -> str:
        """Format retrieved evidence for prompt context"""
        
        if not context.retrieved_results:
            return "No relevant evidence found."
        
        evidence_sections = []
        
        for i, ranked_result in enumerate(context.retrieved_results, 1):
            result = ranked_result.result
            
            # Format individual evidence piece
            evidence_piece = f"""
Evidence {i} (Relevance: {ranked_result.ranking_score:.2f}):
- Type: {result.metadata.data_type}
- Source: {result.metadata.source_file or 'Unknown'}
- Timestamp: {result.metadata.timestamp or 'Unknown'}
- Content: {result.text[:500]}{'...' if len(result.text) > 500 else ''}
"""
            
            # Add entity information if available
            if result.metadata.entities:
                entities_str = []
                for entity_type, entity_list in result.metadata.entities.items():
                    entities_str.append(f"{entity_type}: {', '.join(entity_list[:3])}")
                evidence_piece += f"- Entities: {'; '.join(entities_str)}\n"
            
            # Add participants if available
            if result.metadata.participants:
                evidence_piece += f"- Participants: {', '.join(result.metadata.participants)}\n"
            
            evidence_sections.append(evidence_piece)
        
        return "\n".join(evidence_sections)

class AdvancedResponseGenerator:
    """
    Advanced response generation system with multi-LLM support and forensic specialization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.prompt_engine = ForensicPromptEngine(config)
        self.primary_provider = config.get('primary_provider', 'openai')
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers"""
        
        # OpenAI provider
        if self.config.get('openai_config'):
            try:
                self.providers['openai'] = OpenAIProvider(self.config['openai_config'])
                if self.providers['openai'].is_available():
                    logger.info("Initialized OpenAI provider")
                else:
                    del self.providers['openai']
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI provider: {e}")
        
        # Google provider
        if self.config.get('google_config'):
            try:
                self.providers['google'] = GoogleProvider(self.config['google_config'])
                if self.providers['google'].is_available():
                    logger.info("Initialized Google provider")
                else:
                    del self.providers['google']
            except Exception as e:
                logger.warning(f"Failed to initialize Google provider: {e}")
        
        # HuggingFace provider
        if self.config.get('huggingface_config'):
            try:
                self.providers['huggingface'] = HuggingFaceProvider(self.config['huggingface_config'])
                if self.providers['huggingface'].is_available():
                    logger.info("Initialized HuggingFace provider")
                else:
                    del self.providers['huggingface']
            except Exception as e:
                logger.warning(f"Failed to initialize HuggingFace provider: {e}")
        
        if not self.providers:
            logger.warning("No LLM providers available")
        
        # Adjust primary provider if not available
        if self.primary_provider not in self.providers and self.providers:
            self.primary_provider = list(self.providers.keys())[0]
            logger.info(f"Primary provider changed to: {self.primary_provider}")
    
    def generate_response(
        self,
        query: str,
        context: GenerationContext,
        provider_preference: Optional[str] = None
    ) -> GeneratedResponse:
        """
        Generate response using the best available provider
        
        Args:
            query: User query
            context: Generation context with retrieved results
            provider_preference: Optional provider preference
        
        Returns:
            Generated response with metadata
        """
        
        # Select provider
        provider_name = provider_preference or self.primary_provider
        if provider_name not in self.providers:
            if self.providers:
                provider_name = list(self.providers.keys())[0]
            else:
                return GeneratedResponse(
                    response_text="No LLM providers available for response generation.",
                    confidence_score=0.0,
                    sources_used=[],
                    model_used="none",
                    warnings=["No providers available"]
                )
        
        provider = self.providers[provider_name]
        
        # Determine query type from context
        query_type = getattr(context, 'query_type', 'general')
        if hasattr(context, 'retrieved_results') and context.retrieved_results:
            # Infer query type from result types
            data_types = [r.result.metadata.data_type for r in context.retrieved_results]
            if 'conversation' in data_types:
                query_type = 'communication'
            elif any('call' in dt for dt in data_types):
                query_type = 'timeline'
            elif 'contact' in data_types or 'entity' in data_types:
                query_type = 'entity'
        
        # Create forensic-specific prompt
        formatted_prompt = self.prompt_engine.create_prompt(
            query=query,
            context=context,
            query_type=query_type
        )
        
        # Generate response
        try:
            response = provider.generate_response(formatted_prompt, context)
            
            # Post-process response
            processed_response = self._post_process_response(response, context)
            
            logger.info(f"Generated response using {provider_name} (confidence: {processed_response.confidence_score:.2f})")
            return processed_response
            
        except Exception as e:
            logger.error(f"Response generation failed with {provider_name}: {e}")
            
            # Try fallback provider
            for fallback_name, fallback_provider in self.providers.items():
                if fallback_name != provider_name:
                    try:
                        response = fallback_provider.generate_response(formatted_prompt, context)
                        response.warnings = response.warnings or []
                        response.warnings.append(f"Primary provider {provider_name} failed, used fallback {fallback_name}")
                        return self._post_process_response(response, context)
                    except Exception as fallback_error:
                        logger.error(f"Fallback provider {fallback_name} also failed: {fallback_error}")
            
            # All providers failed
            return GeneratedResponse(
                response_text="Unable to generate response due to provider failures.",
                confidence_score=0.0,
                sources_used=[],
                model_used="failed",
                warnings=[f"All providers failed: {str(e)}"]
            )
    
    def _post_process_response(
        self,
        response: GeneratedResponse,
        context: GenerationContext
    ) -> GeneratedResponse:
        """Post-process generated response"""
        
        # Add source citations if requested
        if context.include_sources and response.sources_used:
            citations = self._format_citations(response.sources_used, context.retrieved_results)
            if citations:
                response.response_text += f"\n\n**Sources:**\n{citations}"
        
        # Add confidence indicator if requested
        if context.include_confidence:
            confidence_indicator = self._format_confidence_indicator(response.confidence_score)
            response.response_text += f"\n\n**Confidence Level:** {confidence_indicator}"
        
        # Limit response length if specified
        if len(response.response_text) > context.max_response_length:
            response.response_text = response.response_text[:context.max_response_length] + "..."
            response.warnings = response.warnings or []
            response.warnings.append("Response truncated due to length limit")
        
        return response
    
    def _format_citations(
        self,
        source_ids: List[str],
        retrieved_results: List[RankedResult]
    ) -> str:
        """Format source citations"""
        
        citations = []
        for i, (source_id, result) in enumerate(zip(source_ids, retrieved_results), 1):
            citation = f"[{i}] {result.result.metadata.data_type}"
            if result.result.metadata.case_id:
                citation += f" (Case: {result.result.metadata.case_id})"
            if result.result.metadata.timestamp:
                citation += f" - {result.result.metadata.timestamp.strftime('%Y-%m-%d %H:%M')}"
            citations.append(citation)
        
        return "\n".join(citations)
    
    def _format_confidence_indicator(self, confidence_score: float) -> str:
        """Format confidence indicator"""
        if confidence_score >= 0.8:
            return f"High ({confidence_score:.1%}) - Response based on strong evidence correlation"
        elif confidence_score >= 0.6:
            return f"Medium ({confidence_score:.1%}) - Response based on moderate evidence correlation"
        elif confidence_score >= 0.4:
            return f"Low ({confidence_score:.1%}) - Response based on limited evidence correlation"
        else:
            return f"Very Low ({confidence_score:.1%}) - Response may require additional verification"
    
    def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get information about available providers"""
        providers_info = []
        for name, provider in self.providers.items():
            info = provider.get_model_info()
            info['name'] = name
            info['available'] = provider.is_available()
            providers_info.append(info)
        
        return providers_info
    
    def generate_multiple_responses(
        self,
        query: str,
        context: GenerationContext,
        providers: Optional[List[str]] = None
    ) -> Dict[str, GeneratedResponse]:
        """Generate responses from multiple providers for comparison"""
        
        if providers is None:
            providers = list(self.providers.keys())
        
        responses = {}
        
        for provider_name in providers:
            if provider_name in self.providers:
                try:
                    response = self.generate_response(query, context, provider_name)
                    responses[provider_name] = response
                except Exception as e:
                    logger.error(f"Failed to generate response with {provider_name}: {e}")
                    responses[provider_name] = GeneratedResponse(
                        response_text=f"Error: {str(e)}",
                        confidence_score=0.0,
                        sources_used=[],
                        model_used=provider_name,
                        warnings=[str(e)]
                    )
        
        return responses