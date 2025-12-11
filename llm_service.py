
from config import settings
from typing import List, Dict
import time
import re
import html
from utils import is_math_question  # ✅ Import from shared utils


class LLMService:
    """Service for LLM completions - supports Gemini Pro + GPT-4o-mini for math"""
    
    def __init__(self):
        self.provider = settings.llm_provider
        
        # Initialize both providers for smart routing
        if self.provider == "gemini":
            import google.generativeai as genai
            if not settings.google_api_key:
                raise Exception("GOOGLE_API_KEY not found in .env file")
            genai.configure(api_key=settings.google_api_key)
            
            # ✅ RENAMED: gemini_flash → gemini_model (since we're using Pro)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
            self.genai = genai  # Store for direct calls
            
            # Initialize OpenAI for math questions
            from openai import OpenAI
            if not settings.openai_api_key:
                raise Exception("OPENAI_API_KEY not found in .env file for math routing")
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
            
            print(f"✓ Using Gemini Pro: gemini-2.5-flash-lite (general questions)")
            print(f"✓ Using GPT-4o-mini: gpt-4o-mini (math questions - FAST mode)")
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.openai_api_key)
            print(f"✓ Using OpenAI LLM: gpt-4o")
    
    def _decode_html_entities(self, text: str) -> str:
        """Decode HTML entities like &lt; to < and &gt; to >"""
        return html.unescape(text)
    
    def _format_math_question(self, user_message: str) -> str:
        """
        Ensure math questions explicitly ask for ANSWER: [number] format
        This is CRITICAL for correct answer extraction
        """
        # If already has the instruction, don't add it again
        if "ANSWER:" in user_message or "answer in the format" in user_message.lower():
            return user_message
        
        # Add the formatting instruction
        return f"""{user_message}

Please solve step by step and provide your answer in the format: ANSWER: [option number]"""
    
    def call_gemini_direct(self, prompt: str, temperature: float = 0, max_tokens: int = 8000) -> str:
        """
        ✅ Direct Gemini call without smart routing
        This is used for non-math questions to avoid false math detection
        """
        if self.provider != "gemini":
            raise Exception("call_gemini_direct() only works with Gemini provider")
        
        try:
            # ✅ FIXED: Add top_k and top_p for deterministic output
            generation_config = self.genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_k=1,  # ✅ CRITICAL: Forces deterministic token selection
                top_p=1.0  # ✅ CRITICAL: Considers all probability mass
            )
            
            # ✅ CRITICAL: Verify config
            print(f"   🌡️ Gemini Config: temp={temperature}, tokens={max_tokens}, top_p=1.0, top_k=1")
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
            
            # ✅ CHANGED: gemini_flash → gemini_model
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Check if blocked by safety at prompt level
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                    print(f"   ⚠️ Gemini blocked prompt (reason: {response.prompt_feedback.block_reason})")
                    return "NEEDS_WEB_SEARCH"
            
            if not response.candidates or len(response.candidates) == 0:
                print("   ⚠️ Gemini returned no candidates")
                return "NEEDS_WEB_SEARCH"
            
            candidate = response.candidates[0]
            
            if hasattr(candidate, 'finish_reason'):
                if candidate.finish_reason in [3, 4]:
                    reason_name = "SAFETY" if candidate.finish_reason == 3 else "RECITATION"
                    print(f"   ⚠️ Gemini blocked response (finish_reason: {reason_name})")
                    return "NEEDS_WEB_SEARCH"
            
            if hasattr(response, 'text') and response.text:
                return response.text
            elif candidate.content and candidate.content.parts:
                text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                if text_parts:
                    return "".join(text_parts)
            
            print("   ⚠️ Gemini returned empty response")
            return "NEEDS_WEB_SEARCH"
                
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['safety', 'blocked', 'harmful', 'policy', 'candidate']):
                print(f"   ⚠️ Gemini safety error: {str(e)[:100]}")
                return "NEEDS_WEB_SEARCH"
            else:
                print(f"   ✗ Gemini error: {str(e)[:200]}")
                raise
    
    def chat_completion(self, system_message: str, user_message: str, temperature: float = 0, max_tokens: int = 2500, bypass_routing: bool = False) -> str:
        """Get chat completion with smart routing: GPT-4o-mini for math, Gemini Pro for others
        
        Args:
            system_message: System prompt
            user_message: User query
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum response length
            bypass_routing: If True, skip smart routing and use Gemini directly (prevents false math detection)
        """
        
        # Decode HTML entities in the user message
        user_message = self._decode_html_entities(user_message)
        
        # ✅ BYPASS ROUTING: Use Gemini directly without math detection
        # This is critical for prompts that contain math keywords in instructions
        # but are NOT actually math questions (e.g., "ব্যাকরণ/সাহিত্য/ইতিহাস/গণিত")
        if bypass_routing and self.provider == "gemini":
            print(f"   🔧 Bypassing smart routing - using Gemini directly")
            combined_prompt = f"{system_message}\n\n{user_message}" if system_message else user_message
            return self.call_gemini_direct(combined_prompt, temperature, max_tokens)
        
        # ✅ SMART ROUTING - Now uses shared function from utils
        is_math = is_math_question(user_message)  # ← Changed from self._is_math_question()
        
        # TIER 1: ALL Math → GPT-4o-mini (⚡ OPTIMIZED FOR SPEED)
        if is_math and self.provider == "gemini":
            print(f"   🎯 MATH detected - routing to GPT-4o-mini (⚡ FAST mode)")
            
            # ✅ Format question to explicitly ask for ANSWER: [number]
            formatted_message = self._format_math_question(user_message)
            
            # ⚡ OPTIMIZED: Shorter system prompt for speed
            math_system_prompt = "You are a math tutor. Be concise."
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    max_tokens=max_tokens,
                    timeout=10.0,  # ⚡ 10 second timeout for speed
                    messages=[
                        {"role": "system", "content": math_system_prompt},
                        {"role": "user", "content": formatted_message}
                    ]
                )
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"   ✗ GPT-4o-mini error: {str(e)[:200]}")
                raise
        
        # TIER 2: Everything else → Gemini Pro
        if self.provider == "gemini":
            if not is_math:
                print(f"   💬 Non-math question - using Gemini Pro")
            
            combined_prompt = f"{system_message}\n\n{user_message}" if system_message else user_message
            
            # ✅ FIXED: Add top_k and top_p for deterministic output
            generation_config = self.genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_k=1,  # ✅ CRITICAL: Forces deterministic token selection
                top_p=1.0  # ✅ CRITICAL: Considers all probability mass
            )
            
            # ✅ CRITICAL: Verify config
            print(f"   🌡️ Gemini Config: temp={temperature}, tokens={max_tokens}, top_p=1.0, top_k=1")
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
            
            try:
                # ✅ CHANGED: gemini_flash → gemini_model
                response = self.gemini_model.generate_content(
                    combined_prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                # Check if blocked by safety at prompt level
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                        print(f"   ⚠️ Gemini blocked prompt (reason: {response.prompt_feedback.block_reason})")
                        return "NEEDS_WEB_SEARCH"
                
                if not response.candidates or len(response.candidates) == 0:
                    print("   ⚠️ Gemini returned no candidates")
                    return "NEEDS_WEB_SEARCH"
                
                candidate = response.candidates[0]
                
                if hasattr(candidate, 'finish_reason'):
                    if candidate.finish_reason in [3, 4]:
                        reason_name = "SAFETY" if candidate.finish_reason == 3 else "RECITATION"
                        print(f"   ⚠️ Gemini blocked response (finish_reason: {reason_name})")
                        return "NEEDS_WEB_SEARCH"
                
                if hasattr(response, 'text') and response.text:
                    return response.text
                elif candidate.content and candidate.content.parts:
                    text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                    if text_parts:
                        return "".join(text_parts)
                
                print("   ⚠️ Gemini returned empty response")
                return "NEEDS_WEB_SEARCH"
                    
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['safety', 'blocked', 'harmful', 'policy', 'candidate']):
                    print(f"   ⚠️ Gemini safety error: {str(e)[:100]}")
                    return "NEEDS_WEB_SEARCH"
                else:
                    print(f"   ✗ Gemini error: {str(e)[:200]}")
                    raise
            
        else:
            # OpenAI as primary provider
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.choices[0].message.content


def get_llm_service() -> LLMService:
    """Get singleton LLM service instance"""
    return LLMService()