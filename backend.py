
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import re
from vector_db import get_vector_db, EmbeddingService
from config import settings
from llm_service import get_llm_service


app = FastAPI(title="Fact Checker & MCQ Validator API")


app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)


vector_db = get_vector_db()
embedding_service = EmbeddingService()
llm_service = get_llm_service()


COLLECTION_NAME = "fact_check_questions"
NEWS_COLLECTION_NAME = "news_articles"


def call_llm(system_message: str, user_message: str, temperature: float = 0, max_tokens: int = 2500) -> str:
   """Helper function to call configured LLM (OpenAI or Gemini)"""
   return llm_service.chat_completion(system_message, user_message, temperature, max_tokens)


class FactCheckRequest(BaseModel):
   question: str
   answer: str
   option1: str
   option2: str
   option3: str
   option4: str
   option5: Optional[str] = None
   explanation: Optional[str] = None
   explain: Optional[str] = None
   language: Optional[str] = "auto"
  
   def get_explanation(self) -> Optional[str]:
       """Get explanation from either 'explanation' or 'explain' field"""
       return self.explanation or self.explain


class OptionValidation(BaseModel):
   feedback: str = ""


class OptionsValidation(BaseModel):
   option1: OptionValidation
   option2: OptionValidation
   option3: OptionValidation
   option4: OptionValidation
   option5: OptionValidation
   options_consistency_valid: bool
   feedback: str = ""


class FactCheckResponse(BaseModel):
   question_valid: bool
   feedback: str = ""
   logical_valid: bool
   options: OptionsValidation
   explanation_valid: bool
   given_answer_valid: bool
   final_answer: str


@app.get("/")
async def root():
   return {"message": "Fact Checker & MCQ Validator API", "status": "online", "llm_provider": settings.llm_provider}


@app.get("/health")
async def health():
   try:
       test_embedding = embedding_service.embed_query("test")
       return {"status": "healthy", "llm_provider": settings.llm_provider, "embedding_type": settings.embedding_type}
   except Exception as e:
       return {"status": "unhealthy", "error": str(e)}


def detect_language(text: str) -> str:
   bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
   total_chars = len([c for c in text if c.isalpha()])
   if total_chars == 0:
       return "en"
   return "bn" if (bengali_chars / total_chars) > 0.3 else "en"


def clean_json(content: str) -> str:
   content = re.sub(r'```json\s*', '', content)
   content = re.sub(r'```\s*', '', content).strip()
   match = re.search(r'\{.*\}', content, re.DOTALL)
   return match.group(0) if match else content


def normalize_answer(answer: str) -> str:
   """
   ✅ PERFECT: Normalize answer by removing option prefixes and extra whitespace
   Handles: "ক)", "খ)", "গ)", "ঘ)", "a)", "b)", "c)", "d)", "1)", "2)", etc.
   """
   if not answer:
       return ""
  
   patterns = [
       r'^[ক-ঙ]\)\s*',
       r'^[a-eA-E]\)\s*',
       r'^[1-5]\)\s*',
       r'^[ক-ঙ]\s*।\s*',
       r'^[a-eA-E]\s*\.\s*',
       r'^[1-5]\s*\.\s*',
   ]
  
   normalized = answer.strip()
   for pattern in patterns:
       normalized = re.sub(pattern, '', normalized)
  
   normalized = ' '.join(normalized.split())
  
   return normalized.strip().lower()


def detect_duplicates(options: List[str]) -> tuple:
   """
   ✅ CORRECT: Strictly detect duplicate options using Python comparison
   Returns (has_duplicates: bool, feedback: str)
   """
   non_empty_options = [(i+1, opt.strip().lower()) for i, opt in enumerate(options) if opt and opt.strip()]
  
   if len(non_empty_options) < 2:
       return False, ""
  
   duplicates = {}
   for i, (idx1, opt1) in enumerate(non_empty_options):
       for idx2, opt2 in non_empty_options[i+1:]:
           if opt1 == opt2:
               if opt1 not in duplicates:
                   duplicates[opt1] = [idx1]
               if idx2 not in duplicates[opt1]:
                   duplicates[opt1].append(idx2)
  
   if not duplicates:
       return False, ""
  
   feedback_parts = []
   for value, indices in duplicates.items():
       if len(indices) > 1:
           options_str = " and ".join([f"Option {idx}" for idx in indices])
           feedback_parts.append(f"{options_str} are duplicates (both have '{value}')")
  
   return True, ". ".join(feedback_parts) + "."


def validate_explanation_correctness(explanation: str, question: str, answer: str, options: List[str]) -> Dict[str, Any]:
   """
   ✅ IMPROVED: Validate if explanation is correct using configured LLM
   """
   try:
       print("\n🔍 Validating explanation correctness...")
      
       validation_system = """You are an expert fact-checker, mathematician, and educator. Your job is to validate if explanations are correct.


⚠️ CRITICAL INSTRUCTIONS:


1. IGNORE FORMATTING ISSUES:
  - HTML entities (&times;, &there4;, &nbsp;, etc.) - IGNORE THESE
  - Missing spaces between words - FOCUS ON CONTENT
  - Bengali/English mixed text - THIS IS OK
  - Poor formatting - FOCUS ON THE ACTUAL CONTENT


2. WHAT TO CHECK:
  For Math Questions:
  - Are calculations correct? (verify every step)
  - Is the formula right?
  - Does the final answer match?
  - Is the logic sound?
 
  For Non-Math Questions:
  - Are facts accurate?
  - Is reasoning logical?
  - Does explanation support the answer?
  - Is information correct?


3. BE REASONABLE:
  - If explanation shows CORRECT work, mark it VALID
  - If calculations are RIGHT, mark it VALID
  - If facts are CORRECT, mark it VALID
  - Only mark INVALID if there are ACTUAL ERRORS in content


EXAMPLES:


✅ VALID - Correct math despite HTML entities:
Explanation: "a=5সাধারণ অন্তর, d=3n=10সমান্তর ধারার n তম পদ = a+(n-1)d&there4; 10 তম পদ = 5+(10-1)3=5+(9&times;3)=32"
Analysis: a=5, d=3, 10th term = 5+9×3 = 32 ✓ ALL CORRECT
→ is_valid: TRUE


✅ VALID - Correct despite formatting:
Explanation: "১ম পদ, a = 5সাধারণ অন্তর, d = 8 - 5 = 3"
Analysis: First term = 5, common difference = 3 ✓ CORRECT
→ is_valid: TRUE


❌ INVALID - Wrong calculations:
Explanation: "√25 = 10, so 10+20 = 30"
Analysis: √25 = 5 NOT 10 ✗ MATH ERROR
→ is_valid: FALSE


✅ VALID - Correct factual explanation:
Explanation: "Dhaka is the capital of Bangladesh since 1971"
Analysis: Factually correct ✓
→ is_valid: TRUE


Return JSON:
{
   "is_valid": true/false,
   "confidence": 95,
   "reasoning": "explain why valid or what errors found"
}"""
      
       opts = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
      
       validation_user = f"""Question: {question}


Options:
{opts}


Correct Answer: {answer}


Explanation to Validate:
{explanation}


Task: Is this explanation correct?
- Ignore formatting/HTML entities
- Check if calculations/facts are RIGHT
- Check if it supports the answer: {answer}


Return ONLY JSON with is_valid, confidence, and reasoning."""


       response = call_llm(validation_system, validation_user)
       result = json.loads(clean_json(response))
      
       is_valid = result.get('is_valid', False)
       confidence = result.get('confidence', 0)
       reasoning = result.get('reasoning', '')
      
       print(f"  LLM Validation: {'✅ VALID' if is_valid else '❌ INVALID'}")
       print(f"  Confidence: {confidence}%")
       print(f"  Reasoning: {reasoning}")
      
       return {
           'is_valid': is_valid and confidence >= 60,
           'confidence': confidence,
           'reasoning': reasoning
       }
      
   except Exception as e:
       print(f"  ✗ Validation error: {e}")
       return {'is_valid': False, 'confidence': 0, 'reasoning': str(e)}


def validate_structure_only(request: FactCheckRequest) -> Dict[str, Any]:
   """
   ✅ CORRECT: Validates question with reasonable strictness
   """
   try:
       system_msg = """You are a question validator. Check if the question and options are reasonable and understandable.


⚠️ IMPORTANT: Be REASONABLE, not overly strict. Many questions are translations from Bengali and may have minor grammatical imperfections but are still perfectly valid and understandable.


QUESTION VALIDATION:
Mark as INVALID only if:
- Question is completely nonsensical or gibberish
- Question has severe logical contradictions
- Question is impossible to understand
- Question is incomplete to the point of being unanswerable


✅ Mark as VALID if:
- Question is understandable despite minor grammar issues
- Question makes logical sense even if phrasing could be better
- Question is clear enough to answer
- Bengali/translation questions with acceptable grammar


LOGICAL VALIDATION:
Mark logical_valid as FALSE only if:
- Severe logical contradictions
- Options are completely wrong type
- Question-option combination makes no sense at all


✅ Mark as VALID if:
- Options are appropriate type for the question
- Minor mismatches are acceptable
- Question and options work together reasonably


OPTION VALIDATION:
Mark options as INVALID only if:
- Completely meaningless gibberish
- Obviously fake placeholder text
- Totally wrong type for question


✅ Mark as VALID if:
- Options make sense for the question
- Options are readable and meaningful
- Minor formatting issues are acceptable


BE REASONABLE. If a human can understand it, mark it VALID.


Return JSON:
{
   "question_valid": true/false,
   "question_feedback": "only if invalid, explain briefly",
   "logical_valid": true/false,
   "logical_feedback": "only if invalid, explain briefly",
   "option1_valid": true/false,
   "option1_feedback": "only if invalid",
   "option2_valid": true/false,
   "option2_feedback": "only if invalid",
   "option3_valid": true/false,
   "option3_feedback": "only if invalid",
   "option4_valid": true/false,
   "option4_feedback": "only if invalid",
   "explanation_valid": true/false,
   "explanation_feedback": ""
}


Return ONLY JSON."""


       explanation_text = request.get_explanation()
       has_exp = bool(explanation_text and explanation_text.strip())
      
       human_msg = f"""Validate this question reasonably (not too strict):


Question: {request.question}
Option 1: {request.option1}
Option 2: {request.option2}
Option 3: {request.option3}
Option 4: {request.option4}
Option 5: {request.option5}
Explanation: {explanation_text if has_exp else 'NOT PROVIDED'}


Check:
1. Is the question understandable? (minor grammar issues OK)
2. Are options appropriate? (minor issues OK)
3. Does it make reasonable sense? (don't be overly strict)


Be REASONABLE. Mark as valid if humans can understand it.


Return JSON."""


       response = call_llm(system_msg, human_msg)
       result = json.loads(clean_json(response))
      
       if not has_exp:
           result['explanation_valid'] = False
           result['explanation_feedback'] = "Not provided"
      
       options = [request.option1, request.option2, request.option3, request.option4, request.option5]
       has_duplicates, duplicate_feedback = detect_duplicates(options)
      
       result['options_consistency_valid'] = not has_duplicates
       result['options_consistency_feedback'] = duplicate_feedback
      
       return result
      
   except Exception as e:
       print(f"✗ Validation error: {e}")
       return {
           "question_valid": True, "question_feedback": "",
           "logical_valid": True, "logical_feedback": "",
           "option1_valid": True, "option1_feedback": "",
           "option2_valid": True, "option2_feedback": "",
           "option3_valid": True, "option3_feedback": "",
           "option4_valid": True, "option4_feedback": "",
           "option5_valid": True, "option5_feedback": "",
           "options_consistency_valid": True, "options_consistency_feedback": "",
           "explanation_valid": False, "explanation_feedback": "Not provided"
       }


def get_answer_from_explanation(explanation: str, question: str, options: List[str]) -> Optional[str]:
   """
   ✅ IMPROVED: Extract answer from explanation
   """
   try:
       print("\n📝 Extracting answer from explanation...")
      
       system_msg = """You are an expert at analyzing explanations to determine the correct answer.


Your job: Read the explanation and find which option it concludes is correct.


IMPORTANT:
- IGNORE HTML entities (&times;, &there4;, &nbsp;)
- IGNORE formatting issues
- IGNORE missing spaces
- FOCUS on the FINAL ANSWER/CONCLUSION in the explanation


Look for:
- Final calculated value
- Concluding statement
- What the explanation says is correct
- The answer shown at the end


Return ONLY JSON:
{
   "answer": "the answer as EXACT TEXT from options list",
   "confidence": 90,
   "reasoning": "where/how you found this answer"
}"""
      
       opts = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
      
       user_msg = f"""Question: {question}


Options:
{opts}


Explanation:
{explanation}


What is the FINAL ANSWER according to this explanation?
- Look at the conclusion/final calculation
- Ignore formatting issues
- Return the answer value


Return ONLY JSON."""


       response = call_llm(system_msg, user_msg)
       result = json.loads(clean_json(response))
      
       explanation_answer = result.get('answer', '').strip()
       confidence = result.get('confidence', 0)
       reasoning = result.get('reasoning', '')
      
       if not explanation_answer or confidence < 50:
           print(f"  ✗ Could not extract answer (confidence: {confidence}%)")
           return None
      
       print(f"  ✓ Extracted answer: '{explanation_answer}'")
       print(f"  ✓ Confidence: {confidence}%")
       print(f"  ✓ Reasoning: {reasoning}")
       return explanation_answer
      
   except Exception as e:
       print(f"✗ Explanation extraction error: {e}")
       return None


def get_answer_from_dataset(question: str, options: List[str]) -> Optional[str]:
   """
   ✅ CORRECT: Find SAME/SIMILAR question in dataset and return its answer
   """
   try:
       print("\n💾 Searching dataset for same/similar question...")
      
       query_emb = embedding_service.embed_query(question)
       results = vector_db.search(COLLECTION_NAME, query_emb, top_k=10)
      
       if not results:
           print("✗ No results from dataset")
           return None
      
       print(f"✓ Found {len(results)} similar questions")
      
       best = max(results, key=lambda x: x.get('score', 0))
       similarity = best.get('score', 0)
       matched_question = best.get('question', '')
      
       print(f"  Best match:")
       print(f"  Similarity: {similarity:.4f}")
       print(f"  Question: {matched_question[:100]}...")
      
       if similarity >= 0.85:
           print(f"  ✓ HIGH similarity - This looks like the SAME question")
           try:
               stored_options = json.loads(best.get('options', '{}'))
               answer_num = best.get('answer')
               stored_explanation = best.get('explanation', '').strip()
              
               dataset_options = [
                   stored_options.get('option1', '').strip(),
                   stored_options.get('option2', '').strip(),
                   stored_options.get('option3', '').strip(),
                   stored_options.get('option4', '').strip()
               ]
              
               matching_options = 0
               for curr_opt in options:
                   curr_opt_norm = normalize_answer(curr_opt)
                   for ds_opt in dataset_options:
                       ds_opt_norm = normalize_answer(ds_opt)
                       if curr_opt_norm and ds_opt_norm and curr_opt_norm == ds_opt_norm:
                           matching_options += 1
                           break
              
               print(f"  Options matching: {matching_options}/{len(options)}")
              
               # ✅ RELAXED: For perfect/near-perfect matches (>= 0.95), require only 2/4 options
               # For lower similarity (0.85-0.95), require 3/4 options
               required_matches = 2 if similarity >= 0.95 else 3
               
               if matching_options < required_matches:
                   print(f"  ✗ Options don't match well enough ({matching_options}/{len(options)}, need {required_matches})")
                   print(f"  → Similarity: {similarity:.4f}, Required matches: {required_matches}")
                   print(f"  → This might be a similar but different question")
                   print(f"  → Will try LLM Knowledge Base instead")
                   return None
              
               print(f"  ✓ Options match well - This is definitely the same question")
              
               if stored_explanation:
                   print("  ✓ Explanation found in dataset")
                   print(f"  Explanation: {stored_explanation[:100]}...")
                  
                   dataset_options_full = [
                       stored_options.get('option1', ''),
                       stored_options.get('option2', ''),
                       stored_options.get('option3', ''),
                       stored_options.get('option4', '')
                   ]
                  
                   try:
                       system_msg = """Extract the correct answer from this explanation.
Return ONLY JSON: {"answer": "answer text", "confidence": 90}"""
                      
                       opts_text = "\n".join([f"{i+1}. {o}" for i, o in enumerate(dataset_options_full) if o])
                       user_msg = f"Question: {matched_question}\n\nOptions:\n{opts_text}\n\nExplanation: {stored_explanation}\n\nReturn ONLY JSON."
                      
                       response = call_llm(system_msg, user_msg)
                       result = json.loads(clean_json(response))
                       extracted_answer = result.get('answer', '').strip()
                      
                       if extracted_answer:
                           for opt in options:
                               if opt.strip().lower() == extracted_answer.strip().lower():
                                   print(f"  ✓ Answer from dataset explanation: '{opt}'")
                                   return opt
                          
                           print(f"  ✓ Answer from dataset explanation: '{extracted_answer}'")
                           return extracted_answer
                   except:
                       pass
              
               if answer_num:
                   answer_text = stored_options.get(f'option{answer_num}', '').strip()
                  
                   if answer_text:
                       for opt in options:
                           if opt.strip().lower() == answer_text.strip().lower():
                               print(f"  ✓ Answer from dataset (option {answer_num}): '{opt}'")
                               return opt
                      
                       print(f"  ✓ Answer from dataset (option {answer_num}): '{answer_text}'")
                       return answer_text
                   else:
                       print(f"  ✗ Could not get text for option {answer_num}")
               else:
                   print("  ✗ No answer number in dataset")
              
           except Exception as e:
               print(f"  ✗ Error extracting from dataset: {e}")
               import traceback
               traceback.print_exc()
       else:
           print(f"  ✗ Similarity too low ({similarity:.4f} < 0.85)")
           print(f"  → This is NOT the same question")
           print(f"  → Will try LLM Knowledge Base instead")
      
       return None
      
   except Exception as e:
       print(f"✗ Dataset error: {e}")
       import traceback
       traceback.print_exc()
       return None


def get_answer_from_llm_knowledge(question: str, options: List[str]) -> Optional[str]:
    """
    ✅ SIMPLE & EFFECTIVE: Uses straightforward prompt that works consistently
    """
    max_attempts = 2
    
    for attempt in range(max_attempts):
        try:
            if attempt > 0:
                print(f"\n   🔄 Retry attempt {attempt + 1}/{max_attempts}")
            
            print(f"\n🧠 Asking {settings.llm_provider.upper()} Knowledge Base...")
           
            # Check for "all of the above" option
            has_all_option = False
            all_option_text = None
            for opt in options:
                if opt and opt.strip().lower() in ['সবগুলোই', 'সবগুলো', 'all of the above', 'all of these', 'all above']:
                    has_all_option = True
                    all_option_text = opt.strip()
                    break
           
            # Format options
            options_text = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
            
            # ✅ SIMPLE PROMPT - Gets straight to the point
            if has_all_option:
                prompt = f"""You are an expert validator for academic questions. Analyze and determine the correct answer.

Question: {question}

Options:
{options_text}

⚠️ SPECIAL: Option "{all_option_text}" is present. You MUST check EACH option individually.

Instructions:
1. Identify question category (ব্যাকরণ/সাহিত্য/ইতিহাস/গণিত)
2. Apply appropriate reasoning (grammar rules, calculations, etc.)
3. Check each option:
   - Option 1: CORRECT/INCORRECT - [why]
   - Option 2: CORRECT/INCORRECT - [why]
   - Option 3: CORRECT/INCORRECT - [why]
4. Determine answer:
   - If ALL options CORRECT → Answer is "{all_option_text}"
   - If even ONE INCORRECT → Answer is NOT "{all_option_text}"

Respond clearly:
- Category: [type]
- Correct Answer: [number and text]
- সঠিক উত্তর: [answer]

If you don't have reliable information, respond only: "NEEDS_WEB_SEARCH" """
            else:
                prompt = f"""You are an expert validator for academic questions. Analyze and determine the correct answer.

Question: {question}

Options:
{options_text}

Instructions:
1. Identify question category (ব্যাকরণ/সাহিত্য/ইতিহাস/গণিত)
2. Apply appropriate reasoning (grammar rules, calculations, etc.)
3. Determine correct answer
4. Provide clear justification

Respond clearly:
- Category: [type]
- Correct Answer: [number and text]
- সঠিক উত্তর: [answer]

If you don't have reliable information, respond only: "NEEDS_WEB_SEARCH" """

            print(f"   Using: {settings.llm_provider}")
            if has_all_option:
                print(f"   ⚠️ SPECIAL MODE: '{all_option_text}' option detected")
           
            # Call LLM with empty system message (all info in user prompt)
            result_text = call_llm("", prompt, temperature=0, max_tokens=8000)
           
            print(f"   ✓ Got response (length: {len(result_text)} chars)")
            
            # ✅ Check if response is complete
            if len(result_text) < 300:
                print(f"   ⚠️ Response too short ({len(result_text)} chars)")
                if attempt < max_attempts - 1:
                    continue
                else:
                    return None
            
            print(f"   📝 Response preview: {result_text[:500]}...")
            print()
           
            # Check for web search signal
            if "NEEDS_WEB_SEARCH" in result_text:
                print(f"   ⚠ LLM indicated it needs web search")
                return None
           
            # Check for uncertainty
            no_info_patterns = [
                "I don't have information",
                "I cannot determine",
                "আমার কাছে তথ্য নেই",
                "I'm not certain",
            ]
            
            for pattern in no_info_patterns:
                if pattern.lower() in result_text.lower():
                    print(f"   ⚠ LLM doesn't have reliable information")
                    return None
           
            # ✅ Extract answer - Multiple strategies
            
            # Strategy 1: Look for explicit patterns
            explicit_patterns = [
                r'সঠিক উত্তর:\s*\d*\.?\s*(.+?)(?:\n|$)',
                r'Correct Answer:\s*\d*\.?\s*(.+?)(?:\n|$)',
                r'উত্তর:\s*\d*\.?\s*(.+?)(?:\n|$)',
            ]
            
            for pattern in explicit_patterns:
                match = re.search(pattern, result_text, re.IGNORECASE | re.UNICODE)
                if match:
                    potential_answer = match.group(1).strip()
                    potential_answer = re.sub(r'^\d+\.?\s*', '', potential_answer)
                    potential_answer = potential_answer.rstrip('।').rstrip('.').strip()
                    potential_answer = potential_answer.split('\n')[0].strip()
                    
                    print(f"   🎯 Found answer via pattern: '{potential_answer}'")
                    
                    # Match against options
                    for opt in options:
                        if not opt:
                            continue
                        opt_clean = opt.strip()
                        
                        if (opt_clean.lower() == potential_answer.lower() or 
                            potential_answer.lower() in opt_clean.lower() or 
                            opt_clean.lower() in potential_answer.lower()):
                            print(f"✓ LLM Knowledge ({settings.llm_provider}): '{opt_clean}'")
                            return opt_clean
            
            # Strategy 2: Scoring method
            print("   ⚠ No explicit answer found, using scoring...")
            
            best_match = None
            best_score = 0
            
            for i, opt in enumerate(options):
                if not opt:
                    continue
                    
                opt_clean = opt.strip()
                score = 0
                
                # Check for "all options correct" phrases
                if has_all_option and opt_clean.lower() in ['সবগুলোই', 'সবগুলো', 'all of the above']:
                    all_correct_phrases = [
                        "all options are correct",
                        "সব অপশনই সঠিক",
                        "all correct",
                    ]
                    for phrase in all_correct_phrases:
                        if phrase.lower() in result_text.lower():
                            score += 50
                            break
                
                # High score for answer indicators
                if f"Correct Answer: {i+1}" in result_text or f"সঠিক উত্তর: {opt_clean}" in result_text:
                    score += 50
                
                # Medium score for appearing in last 400 chars
                if opt_clean in result_text[-400:]:
                    score += 20
                
                # Low score for general presence
                if opt_clean in result_text:
                    score += 5
                
                print(f"   Option {i+1} ('{opt_clean}'): {score} points")
                
                if score > best_score:
                    best_score = score
                    best_match = opt_clean
            
            if best_match and best_score >= 20:
                print(f"✓ LLM Knowledge ({settings.llm_provider}): '{best_match}' (score: {best_score})")
                return best_match
            
            print(f"   ⚠ Could not extract answer (best score: {best_score})")
            
            if attempt >= max_attempts - 1:
                return None
           
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            if attempt >= max_attempts - 1:
                return None
   
    return None


def get_answer_from_openai_web_search(question: str, options: List[str]) -> Optional[str]:
    """
    ✅ SOURCE 3: OpenAI Web Search (Fallback after trusted news)
    
    Uses gpt-4o-mini-search-preview model with built-in web search capability
    """
    try:
        from openai import OpenAI
        print("\n🌐 SOURCE 3: OpenAI Web Search (Real-time Internet Search)...")
        
        opts_formatted = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
        
        search_prompt = f"""You are answering a quiz question using ONLY verified authoritative sources.

Question: {question}
Options:
{opts_formatted}

CRITICAL INSTRUCTIONS FOR BENGALI LANGUAGE/GRAMMAR QUESTIONS:
- For বাংলা ব্যাকরণ (Bengali grammar) questions, prioritize:
  1. NCTB textbooks (National Curriculum and Textbook Board)
  2. Bengali grammar books by established authors (হায়াৎ মামুদ, মুনীর চৌধুরী)
  3. Academic sources (.edu.bd domains)
  4. Established Bengali language resources
- DO NOT rely on general web articles or blogs for grammar rules
- Grammar definitions must match NCTB curriculum exactly
- Cross-check definitions across multiple authoritative grammar sources

MANDATORY SEARCH PROCESS:
1. Search 4-6 different TOP TIER sources based on topic
2. For Bengali grammar: Search "NCTB বাংলা ব্যাকরণ" + question topic
3. Cross-reference ALL options against authoritative sources
4. Verify the DEFINITION matches the technical term being asked
5. Count votes: which option appears most in reliable sources
6. Choose the option with highest source agreement (minimum 3 sources)

AUTHORITATIVE SOURCES BY TOPIC:
- Bengali grammar/language: NCTB textbooks, established grammar books, .edu.bd sites
- Bangladesh news/events: Prothom Alo, Daily Star, bdnews24, government sites
- International news: Reuters, AP, BBC, CNN, official statements
- Deaths/casualties: Official government reports, UN, verified news agencies
- Historical events: Wikipedia (cross-check), Britannica, academic sources
- Sports: ESPN, official league sites, verified sports media
- Science/health: WHO, CDC, peer-reviewed journals, Nature, Science
- Technology: official documentation, tech news, official company sites
- Politics: official government sites, established news agencies
- Entertainment: IMDB, official announcements, Variety, Hollywood Reporter
- Geography/statistics: World Bank, UN data, official census
- Business/Economy: Bloomberg, Reuters, Financial Times, official reports
- Education: official university sites, education ministry, verified rankings
- Religion: official religious texts, verified scholarly sources
- Culture/Literature: established publishers, literary databases, verified reviews
- Law/Legal: official government legal sites, verified legal databases

FOR BENGALI GRAMMAR QUESTIONS - SPECIAL INSTRUCTIONS:
1. Understand the question is asking for a TECHNICAL DEFINITION
2. Search for "রূঢ়ি শব্দ definition NCTB" or similar
3. Read the COMPLETE definition from grammar sources
4. Match each option against the definition
5. Select the option that FITS the definition, not just appears in examples

STRICT RULES:
- Never trust a single source
- Ignore blogs, forums, social media claims
- For grammar: NCTB curriculum is the gold standard
- For conflicting data: go with official/government source
- For numbers: use only confirmed figures, never estimates
- Minimum 3 sources must agree before selecting answer
- If topic is unclear, search broader then narrow down
- For niche topics: prioritize domain experts and official organizations

VERIFICATION CHECKLIST:
✓ Is this from a top-tier source for this topic?
✓ For grammar: Does the definition from NCTB match?
✓ Do at least 3 reliable sources confirm this?
✓ Does this match official data (if applicable)?
✓ Are there any contradicting authoritative sources?
✓ Did I verify the DEFINITION, not just find the word in examples?

Return ONLY this JSON:
{{
    "answer": "exact option text confirmed by majority of authoritative sources",
    "confidence": 90,
    "reasoning": "Confirmed by [source1], [source2], [source3]. Cross-checked against [total] sources."
}}
NO markdown blocks. NO extra text. ONLY JSON."""

        print(f"   Using: gpt-4o-mini-search-preview")
        
        openai_client = OpenAI(api_key=settings.openai_api_key)
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini-search-preview",
            messages=[
                {
                    "role": "user",
                    "content": search_prompt
                }
            ],
        )
        
        result_text = response.choices[0].message.content.strip()
        print(f"   ✓ Got search response")
        print(f"   📝 Response: {result_text[:200]}...")
        
        try:
            result_clean = result_text.strip()
            if '```json' in result_clean:
                result_clean = result_clean.split('```json')[1].split('```')[0].strip()
            elif '```' in result_clean:
                result_clean = result_clean.split('```')[1].split('```')[0].strip()
            
            result = json.loads(result_clean)
            
            answer = result.get('answer', '').strip()
            confidence = result.get('confidence', 0)
            reasoning = result.get('reasoning', '')
            
            if answer and confidence >= 60:
                print(f"✓ OpenAI Web Search (gpt-4o-mini-search-preview): '{answer}' ({confidence}%)")
                print(f"  Reasoning: {reasoning}")
                return answer
            else:
                print(f"✗ Low confidence ({confidence}%) or no answer")
                return None
                
        except json.JSONDecodeError as e:
            print(f"  ✗ Could not parse JSON response: {e}")
            print(f"  Response was: {result_text[:300]}")
            
            for opt in options:
                if opt.strip() and opt.strip().lower() in result_text.lower():
                    print(f"✓ OpenAI Web Search (text match): '{opt}'")
                    return opt
            
            return None
            
    except Exception as e:
        print(f"✗ OpenAI Web Search error: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.post("/fact-check", response_model=FactCheckResponse)
async def fact_check(request: FactCheckRequest):
   """
   ✅ PERFECT COMBINED VERSION with configurable LLM (OpenAI or Gemini)
   """
   try:
       lang = detect_language(request.question) if request.language == "auto" else request.language
      
       print(f"\n{'='*80}")
       print("🔍 FACT CHECK REQUEST")
       print(f"{'='*80}")
       print(f"Question: {request.question}")
       print(f"Given Answer: {request.answer}")
       print(f"Language: {lang}")
       print(f"LLM Provider: {settings.llm_provider}")
      
       explanation_text = request.get_explanation()
       has_exp = bool(explanation_text and explanation_text.strip())
       print(f"Explanation: {'PROVIDED' if has_exp else 'NOT PROVIDED'}")
       if has_exp:
           print(f"Explanation text: {explanation_text[:100]}...")
       print(f"{'='*80}\n")
      
       print("STEP 1: Validating question structure, grammar, and options...")
       validation = validate_structure_only(request)
      
       if has_exp:
           validation['explanation_valid'] = True
      
       print("✓ Validation complete\n")
      
       print(f"{'='*80}")
       print("STEP 2: Finding Correct Answer")
       print(f"{'='*80}")
       print("⚠️  INPUT Explanation is ONLY used for validation, NOT for answer extraction")
       print(f"    Answer sources: Dataset → {settings.llm_provider.upper()} Knowledge Base → OpenAI Web Search")
      
       final_answer = None
       options = [request.option1, request.option2, request.option3, request.option4]
      
       # SOURCE 1: Dataset
       print("\n→ SOURCE 1: Dataset")
       final_answer = get_answer_from_dataset(request.question, options)
      
       if final_answer:
           print("✓ SOURCE 1 SUCCESS")
       else:
           print("✗ SOURCE 1 FAILED")
      
       # SOURCE 2: LLM Knowledge Base (Gemini or OpenAI)
       if not final_answer:
           print(f"\n→ SOURCE 2: {settings.llm_provider.upper()} Knowledge Base")
           final_answer = get_answer_from_llm_knowledge(request.question, options)
          
           if final_answer:
               print("✓ SOURCE 2 SUCCESS")
           else:
               print("✗ SOURCE 2 FAILED")
      
       # SOURCE 3: OpenAI Web Search
       if not final_answer:
           print("\n→ SOURCE 3: OpenAI Web Search (Real-time Internet)")
           final_answer = get_answer_from_openai_web_search(request.question, options)
          
           if final_answer:
               print("✓ SOURCE 3 SUCCESS")
           else:
               print("✗ SOURCE 3 FAILED")
      
       if not final_answer:
           final_answer = "Unable to determine the correct answer"
           print("\n❌ ALL SOURCES FAILED")
      
       if has_exp and final_answer and final_answer != "Unable to determine the correct answer":
           print(f"\n{'='*80}")
           print("STEP 3: Validating INPUT Explanation Against CORRECT Answer")
           print(f"{'='*80}")
           print(f"  Correct answer from Dataset/LLM: '{final_answer}'")
           print(f"  Checking if INPUT explanation supports this answer...")
          
           explanation_claims_answer = get_answer_from_explanation(
               explanation_text,
               request.question,
               options
           )
          
           if explanation_claims_answer:
               print(f"  INPUT explanation claims answer is: '{explanation_claims_answer}'")
              
               explanation_normalized = normalize_answer(explanation_claims_answer)
               correct_normalized = normalize_answer(final_answer)
              
               if explanation_normalized == correct_normalized:
                   print(f"  ✓ Explanation answer MATCHES correct answer")
                  
                   exp_validation = validate_explanation_correctness(
                       explanation_text,
                       request.question,
                       final_answer,
                       options
                   )
                   validation['explanation_valid'] = exp_validation['is_valid']
                   if not exp_validation['is_valid']:
                       validation['explanation_feedback'] = exp_validation['reasoning']
               else:
                   print(f"  ✗ Explanation answer DOES NOT match correct answer")
                   print(f"     Explanation says: '{explanation_claims_answer}' (normalized: '{explanation_normalized}')")
                   print(f"     Correct answer is: '{final_answer}' (normalized: '{correct_normalized}')")
                   validation['explanation_valid'] = False
                   validation['explanation_feedback'] = f"Explanation supports wrong answer '{explanation_claims_answer}' but correct answer is '{final_answer}'"
           else:
               print(f"  ✗ Could not extract answer from explanation")
               validation['explanation_valid'] = False
               validation['explanation_feedback'] = "Could not determine what answer the explanation supports"
      
       print(f"\n{'='*80}")
       print("STEP 4: Comparing Given Answer with Correct Answer")
       print(f"{'='*80}")
      
       given_answer_valid = False
       if final_answer and final_answer != "Unable to determine the correct answer":
           given_normalized = normalize_answer(request.answer)
           final_normalized = normalize_answer(final_answer)
          
           print(f"Given (original): '{request.answer}'")
           print(f"Given (normalized): '{given_normalized}'")
           print(f"Correct (original): '{final_answer}'")
           print(f"Correct (normalized): '{final_normalized}'")
          
           given_answer_valid = (given_normalized == final_normalized)
          
           if given_answer_valid:
               print("✓ MATCH: Given answer is CORRECT")
           else:
               print("✗ NO MATCH: Given answer is WRONG")
       else:
           print("✗ Cannot validate: No correct answer found")
           given_answer_valid = False
      
       print(f"\n{'='*80}")
       print("FINAL RESULT")
       print(f"{'='*80}")
       print(f"Correct Answer: {final_answer}")
       print(f"Given Answer: {request.answer}")
       print(f"Given Answer Valid: {given_answer_valid}")
       print(f"Explanation Valid: {validation.get('explanation_valid', False)}")
       print(f"{'='*80}\n")
      
       return FactCheckResponse(
           question_valid=validation.get('question_valid', True),
           feedback=validation.get('question_feedback', '') or '',
           logical_valid=validation.get('logical_valid', True),
           options=OptionsValidation(
               option1=OptionValidation(
                   feedback=validation.get('option1_feedback', '') or ''
               ),
               option2=OptionValidation(
                   feedback=validation.get('option2_feedback', '') or ''
               ),
               option3=OptionValidation(
                   feedback=validation.get('option3_feedback', '') or ''
               ),
               option4=OptionValidation(
                   feedback=validation.get('option4_feedback', '') or ''
               ),
               option5=OptionValidation(
                   feedback=validation.get('option5_feedback', '') or ''
               ),
               options_consistency_valid=validation.get('options_consistency_valid', True),
               feedback=validation.get('options_consistency_feedback', '') or ''
           ),
           explanation_valid=validation.get('explanation_valid', False),
           given_answer_valid=given_answer_valid,
           final_answer=final_answer
       )
      
   except Exception as e:
       print(f"\n{'='*80}")
       print("✗ CRITICAL ERROR")
       print(f"{'='*80}")
       print(f"{e}")
       import traceback
       traceback.print_exc()
       print(f"{'='*80}\n")
       raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
   import uvicorn
   print(f"\n{'='*80}")
   print(f"🚀 Fact Checker & MCQ Validator API - {settings.llm_provider.upper()} Version")
   print(f"{'='*80}")
   print(f"LLM Provider: {settings.llm_provider}")
   print(f"Embedding Type: {settings.embedding_type}")
   print("✅ PERFECT: Answer validation with normalization")
   print("✅ PERFECT: Question/logic validation (reasonable strictness)")
   print("✅ FIXED: Explanation validation (handles HTML entities correctly)")
   print(f"✅ PERFECT: {settings.llm_provider.upper()} Knowledge Base (SIMPLE prompt for consistency)")
   print("="*80)
   print("⚠️  IMPORTANT: INPUT Explanation is ONLY for validation")
   print("    It does NOT determine the answer")
   print("="*80)
   print("Answer Source Priority:")
   print("  1. Dataset (40,000+ questions, similarity 0.85+)")
   print(f"  2. {settings.llm_provider.upper()} Knowledge Base (simple prompt, 8000 tokens)")
   print("  3. OpenAI Web Search (Real-time Internet)")
   print("")
   print("Explanation Validation:")
   print("  - Extracts what answer explanation claims")
   print(f"  - Compares with correct answer from Dataset/{settings.llm_provider.upper()}")
   print("  - Validates math and factual correctness (ignores HTML entities)")
   print("  - Sets explanation_valid field in response")
   print(f"{'='*80}\n")
  
   uvicorn.run(app, host=settings.api_host, port=settings.api_port)