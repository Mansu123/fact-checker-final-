
from config import settings
from typing import List, Dict
import time


class LLMService:
    """Service for LLM completions - supports OpenAI and Gemini"""
    
    def __init__(self):
        self.provider = settings.llm_provider
        
        if self.provider == "gemini":
            import google.generativeai as genai
            if not settings.google_api_key:
                raise Exception("GOOGLE_API_KEY not found in .env file")
            genai.configure(api_key=settings.google_api_key)
            
            # Use gemini-2.0-flash-exp for better handling
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print(f"✓ Using Gemini LLM: gemini-2.5-flash-lite (safety filters disabled)")
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.openai_api_key)
            print(f"✓ Using OpenAI LLM: gpt-4o")
    
    def chat_completion(self, system_message: str, user_message: str, temperature: float = 0, max_tokens: int = 2500) -> str:
        """Get chat completion from configured LLM provider"""
        
        if self.provider == "gemini":
            # Combine system and user messages for Gemini
            combined_prompt = f"{system_message}\n\n{user_message}"
            
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            # ✅ CORRECT: Use proper Gemini safety setting keys
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
            
            try:
                response = self.model.generate_content(
                    combined_prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                # Check if blocked by safety at prompt level
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                        print(f"   ⚠️ Gemini blocked prompt (reason: {response.prompt_feedback.block_reason})")
                        return "NEEDS_WEB_SEARCH"
                
                # Check if response has candidates
                if not response.candidates or len(response.candidates) == 0:
                    print("   ⚠️ Gemini returned no candidates")
                    return "NEEDS_WEB_SEARCH"
                
                candidate = response.candidates[0]
                
                # Check finish reason (using numeric values as they're more reliable)
                if hasattr(candidate, 'finish_reason'):
                    # finish_reason values: 0=UNSPECIFIED, 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
                    if candidate.finish_reason in [3, 4]:  # SAFETY or RECITATION
                        reason_name = "SAFETY" if candidate.finish_reason == 3 else "RECITATION"
                        print(f"   ⚠️ Gemini blocked response (finish_reason: {reason_name}), returning NEEDS_WEB_SEARCH")
                        return "NEEDS_WEB_SEARCH"
                
                # Try to get text
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
                # Check if it's a safety-related error
                if any(keyword in error_str for keyword in ['safety', 'blocked', 'harmful', 'policy', 'candidate']):
                    print(f"   ⚠️ Gemini safety error: {str(e)[:100]}")
                    return "NEEDS_WEB_SEARCH"
                else:
                    print(f"   ✗ Gemini error: {str(e)[:200]}")
                    raise
            
        else:
            # OpenAI
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