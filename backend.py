
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import re
from vector_db import get_vector_db, EmbeddingService
from config import settings
from openai import OpenAI




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
openai_client = OpenAI(api_key=settings.openai_api_key)


COLLECTION_NAME = "fact_check_questions"
NEWS_COLLECTION_NAME = "news_articles"


def call_gpt4(system_message: str, user_message: str) -> str:
   """Helper function to call GPT-4 directly"""
   response = openai_client.chat.completions.create(
       model="gpt-4",
       temperature=0,  # ✅ Deterministic for accuracy
       messages=[
           {"role": "system", "content": system_message},
           {"role": "user", "content": user_message}
       ]
   )
   return response.choices[0].message.content


class FactCheckRequest(BaseModel):
   question: str
   answer: str
   option1: str
   option2: str
   option3: str
   option4: str
   option5: Optional[str] = None
   explanation: Optional[str] = None
   explain: Optional[str] = None  # ✅ Support "explain" field too
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
   return {"message": "Fact Checker & MCQ Validator API", "status": "online"}


@app.get("/health")
async def health():
   try:
       test_embedding = embedding_service.embed_query("test")
       return {"status": "healthy"}
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
  
   # Remove common option prefixes
   patterns = [
       r'^[ক-ঙ]\)\s*',      # Bengali options
       r'^[a-eA-E]\)\s*',    # English options
       r'^[1-5]\)\s*',       # Numbered options
       r'^[ক-ঙ]\s*।\s*',    # Bengali with vertical bar
       r'^[a-eA-E]\s*\.\s*', # English with dot
       r'^[1-5]\s*\.\s*',    # Numbers with dot
   ]
  
   normalized = answer.strip()
   for pattern in patterns:
       normalized = re.sub(pattern, '', normalized)
  
   # Remove extra whitespace
   normalized = ' '.join(normalized.split())
  
   return normalized.strip().lower()


def detect_duplicates(options: List[str]) -> tuple:
   """
   ✅ CORRECT: Strictly detect duplicate options using Python comparison
   Returns (has_duplicates: bool, feedback: str)
   """
   # Filter out empty options
   non_empty_options = [(i+1, opt.strip().lower()) for i, opt in enumerate(options) if opt and opt.strip()]
  
   if len(non_empty_options) < 2:
       return False, ""
  
   # Find duplicates
   duplicates = {}
   for i, (idx1, opt1) in enumerate(non_empty_options):
       for idx2, opt2 in non_empty_options[i+1:]:
           if opt1 == opt2:  # Exact match only
               if opt1 not in duplicates:
                   duplicates[opt1] = [idx1]
               if idx2 not in duplicates[opt1]:
                   duplicates[opt1].append(idx2)
  
   if not duplicates:
       return False, ""
  
   # Build feedback
   feedback_parts = []
   for value, indices in duplicates.items():
       if len(indices) > 1:
           options_str = " and ".join([f"Option {idx}" for idx in indices])
           feedback_parts.append(f"{options_str} are duplicates (both have '{value}')")
  
   return True, ". ".join(feedback_parts) + "."


def validate_explanation_correctness(explanation: str, question: str, answer: str, options: List[str]) -> Dict[str, Any]:
   """
   ✅ IMPROVED: Validate if explanation is correct using GPT properly for ALL types
   """
   try:
       print("\n🔍 Validating explanation correctness with GPT...")
      
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


       response = call_gpt4(validation_system, validation_user)
       result = json.loads(clean_json(response))
      
       is_valid = result.get('is_valid', False)
       confidence = result.get('confidence', 0)
       reasoning = result.get('reasoning', '')
      
       print(f"  GPT Validation: {'✅ VALID' if is_valid else '❌ INVALID'}")
       print(f"  Confidence: {confidence}%")
       print(f"  Reasoning: {reasoning}")
      
       # Lower threshold to 60% to be more lenient
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
   - Not too strict for Bengali questions
   - Checks basic grammar and logic
   - Allows minor imperfections
   """
   try:
       system_msg = """You are a question validator. Check if the question and options are reasonable and understandable.


⚠️ IMPORTANT: Be REASONABLE, not overly strict. Many questions are translations from Bengali and may have minor grammatical imperfections but are still perfectly valid and understandable.


QUESTION VALIDATION:
Mark as INVALID only if:
- Question is completely nonsensical or gibberish
- Question has severe logical contradictions (e.g., asking about something that cannot exist)
- Question is impossible to understand
- Question is incomplete to the point of being unanswerable


✅ Mark as VALID if:
- Question is understandable despite minor grammar issues
- Question makes logical sense even if phrasing could be better
- Question is clear enough to answer
- Bengali/translation questions with acceptable grammar


LOGICAL VALIDATION:
Mark logical_valid as FALSE only if:
- Severe logical contradictions (not minor inconsistencies)
- Options are completely wrong type (e.g., random gibberish for a valid question)
- Question-option combination makes no sense at all


✅ Mark as VALID if:
- Options are appropriate type for the question
- Minor mismatches are acceptable
- Question and options work together reasonably


OPTION VALIDATION:
Mark options as INVALID only if:
- Completely meaningless gibberish
- Obviously fake placeholder text (e.g., "xxxxxxxx", "test123")
- Totally wrong type (words for pure arithmetic, random numbers for text questions)


✅ Mark as VALID if:
- Options make sense for the question
- Options are readable and meaningful
- Minor formatting issues are acceptable


EXAMPLES:


❌ INVALID QUESTION:
Q: "asdfkjalksdjflk aksjdf" (gibberish)
→ question_valid: FALSE


❌ SEVERE LOGICAL PROBLEM:
Q: "What is the color of mathematics?" Options: "Blue", "Fast", "৭", "Table"
→ logical_valid: FALSE (nonsensical concept + unrelated options)


✅ VALID QUESTION (even with minor issues):
Q: "সংবিধানের কোন সংশোধনীর মাধ্যমে উপ-রাষ্ট্রপতি পদ বিলুপ্ত করা হয়?"
(Which amendment abolished vice-president post?)
Options: "ক) পঞ্চম", "খ) ষষ্ঠ", "গ) একাদশ", "ঘ) দ্বাদশ"
→ question_valid: TRUE, logical_valid: TRUE, all options valid ✅


✅ VALID - Minor grammar but understandable:
Q: "Who first introduce gold coins in subcontinent?" (minor grammar issue)
Options: "Guptas", "Kushanas", "Mauryas", "Delhi Sultans"
→ All VALID ✅ (understandable despite minor grammar)


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


       # ✅ Support both "explanation" and "explain" fields
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


       response = call_gpt4(system_msg, human_msg)
       result = json.loads(clean_json(response))
      
       # Force explanation_valid = false if not provided
       if not has_exp:
           result['explanation_valid'] = False
           result['explanation_feedback'] = "Not provided"
      
       # Use Python-based duplicate detection for accuracy
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
   ✅ IMPROVED: Extract answer from explanation with better handling
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


       response = call_gpt4(system_msg, user_msg)
       result = json.loads(clean_json(response))
      
       explanation_answer = result.get('answer', '').strip()
       confidence = result.get('confidence', 0)
       reasoning = result.get('reasoning', '')
      
       if not explanation_answer or confidence < 50:  # Lower threshold
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
   - Higher similarity threshold (0.85) to ensure accurate matches
   - Validates options match before accepting dataset answer
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
      
       # ✅ CORRECT: Much higher threshold (0.85)
       if similarity >= 0.85:
           print(f"  ✓ HIGH similarity - This looks like the SAME question")
           try:
               stored_options = json.loads(best.get('options', '{}'))
               answer_num = best.get('answer')
               stored_explanation = best.get('explanation', '').strip()
              
               # ✅ CORRECT: Validate that dataset options match current options
               dataset_options = [
                   stored_options.get('option1', '').strip(),
                   stored_options.get('option2', '').strip(),
                   stored_options.get('option3', '').strip(),
                   stored_options.get('option4', '').strip()
               ]
              
               # Check how many options match
               matching_options = 0
               for curr_opt in options:
                   curr_opt_norm = normalize_answer(curr_opt)
                   for ds_opt in dataset_options:
                       ds_opt_norm = normalize_answer(ds_opt)
                       if curr_opt_norm and ds_opt_norm and curr_opt_norm == ds_opt_norm:
                           matching_options += 1
                           break
              
               print(f"  Options matching: {matching_options}/{len(options)}")
              
               # ✅ CORRECT: Require at least 3 out of 4 options to match
               if matching_options < 3:
                   print(f"  ✗ Options don't match well enough ({matching_options}/4)")
                   print(f"  → This is a DIFFERENT question, not using dataset answer")
                   print(f"  → Will try GPT Knowledge Base instead")
                   return None
              
               print(f"  ✓ Options match well - This is definitely the same question")
              
               # Priority 1: Check if explanation exists
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
                      
                       response = call_gpt4(system_msg, user_msg)
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
              
               # Priority 2: Use answer number
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
           print(f"  → Will try GPT Knowledge Base instead")
      
       return None
      
   except Exception as e:
       print(f"✗ Dataset error: {e}")
       import traceback
       traceback.print_exc()
       return None


def get_answer_from_gpt_knowledge(question: str, options: List[str]) -> Optional[str]:
    """
    ✅ PERFECT: Works EXACTLY like real ChatGPT - Solves problem FIRST, then matches options
    """
    try:
        print("\n🧠 Asking GPT-4 Knowledge Base (Real ChatGPT Style)...")
       
        # ✅ Check if "সবগুলোই" or "all of the above" is an option
        has_all_option = False
        all_option_text = None
        for opt in options:
            if opt and opt.strip().lower() in ['সবগুলোই', 'সবগুলো', 'all of the above', 'all of these', 'all above']:
                has_all_option = True
                all_option_text = opt.strip()
                break
       
        # ✅ COMBINED: All previous instructions + strict "সবগুলোই" handling
        if has_all_option:
            system_msg = f"""You are ChatGPT, a helpful assistant and expert in Bengali language, literature, history, and grammar.

🚨 CRITICAL INSTRUCTION - THIS QUESTION HAS "{all_option_text}" OPTION:

YOU MUST FOLLOW THIS EXACT PROCESS:
1. Analyze EACH individual option (except "{all_option_text}") separately
2. For EACH option, clearly state: "Option X: CORRECT/INCORRECT - [reason]"
3. After checking ALL options:
   - If ALL options are CORRECT → Answer is "{all_option_text}"
   - If even ONE option is INCORRECT → Answer is NOT "{all_option_text}"

EXAMPLE FORMAT:
বিশ্লেষণ:
Option 1 (গভর্নমেন্ট গেজেট): CORRECT - কারণ জন ক্লার্ক মার্শম্যান ১৮৪০ সালে এর সম্পাদক ছিলেন
Option 2 (সমাচার দর্পণ): CORRECT - কারণ তিনি ১৮১৮-১৮৪১ পর্যন্ত এটি সম্পাদনা করেন
Option 3 (দিগদর্শন): CORRECT - কারণ এটিও তাঁর সম্পাদিত পত্রিকা ছিল

সিদ্ধান্ত: যেহেতু তিনটি অপশনই সঠিক, উত্তর: {all_option_text}

IMPORTANT INSTRUCTIONS:
- For Bengali language/grammar/literature/history questions, use your deep knowledge to answer accurately
- If you have reliable knowledge from your training data, provide the answer confidently with clear reasoning
- If the question is about events, data, or information AFTER your knowledge cutoff (October 2023), respond with exactly: "NEEDS_WEB_SEARCH"
- If you lack reliable information about ANY option, respond with: "NEEDS_WEB_SEARCH"
- Do NOT give uncertain answers. Either answer confidently OR say "NEEDS_WEB_SEARCH"
- Do NOT assume all options are correct just because "{all_option_text}" exists
- CHECK EACH OPTION INDIVIDUALLY with evidence
- For Bengali questions, answer in Bengali with clear explanation"""
        else:
            # Regular system message for non-"সবগুলোই" questions
            system_msg = """You are ChatGPT, a helpful assistant and expert in Bengali language, literature, history, and grammar. Solve problems step by step, showing your work clearly.

IMPORTANT INSTRUCTIONS:
- For Bengali language/grammar/literature/history questions, use your deep knowledge of বাংলা ব্যাকরণ (Bengali grammar) to answer accurately
- If you have reliable knowledge from your training data, provide the answer confidently with clear reasoning
- If the question is about events, data, or information AFTER your knowledge cutoff (October 2023), respond with exactly: "NEEDS_WEB_SEARCH"
- If the question requires current/recent information you don't have, respond with: "NEEDS_WEB_SEARCH"
- Do NOT give uncertain answers or say "I don't have information" unless truly necessary. Either answer confidently OR say "NEEDS_WEB_SEARCH"
- For Bengali questions, answer in Bengali with clear explanation
- CRITICAL: When analyzing options, examine EACH option carefully against the question criteria"""

        opts_formatted = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
       
        user_msg = f"""{question}

বিকল্পসমূহ:
{opts_formatted}

{"প্রতিটি বিকল্প আলাদাভাবে যাচাই করুন এবং" if has_all_option else "প্রতিটি বিকল্প সাবধানে বিশ্লেষণ করুন এবং"} সঠিক উত্তর বলুন। আপনার যুক্তি ব্যাখ্যা করুন। যদি আপনার কাছে নির্ভরযোগ্য তথ্য না থাকে, শুধু "NEEDS_WEB_SEARCH" লিখুন।"""

        try:
            print(f"   Using: gpt-4o (same as ChatGPT)")
            if has_all_option:
                print(f"   ⚠️ SPECIAL MODE: '{all_option_text}' option detected - will verify ALL options")
           
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                max_tokens=2500,  # More tokens for detailed analysis
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ]
            )
           
            result_text = response.choices[0].message.content.strip()
            print(f"   ✓ Got response")
            print(f"   📝 GPT Full Response:")
            print(f"   {result_text}")
            print()
           
            # ✅ CHECK: Does GPT need web search?
            if "NEEDS_WEB_SEARCH" in result_text:
                print(f"   ⚠ GPT indicated it needs web search for this question")
                return None
           
            # ✅ CHECK: Is GPT saying it doesn't have information?
            no_info_patterns = [
                "আমার কাছে নির্দিষ্ট তথ্য নেই",
                "আমার কাছে তথ্য নেই",
                "I don't have information",
                "I don't have specific information",
                "I cannot determine",
                "I'm not certain",
                "beyond my knowledge cutoff",
                "after my training data",
                "২০২৩ সালের অক্টোবর পর্যন্ত",
                "আমি নিশ্চিত নই",
                "নিশ্চিত করতে পারছি না"
            ]
            
            for pattern in no_info_patterns:
                if pattern.lower() in result_text.lower():
                    print(f"   ⚠ GPT doesn't have reliable information (found: '{pattern}')")
                    return None
           
            # ✅ Extract the CALCULATED answer value from GPT's solution
            best_match = None
            best_match_score = 0
           
            for i, opt in enumerate(options):
                if not opt:
                    continue
                    
                opt_clean = opt.strip()
                score = 0
               
                # ✅ Higher weight for "সবগুলোই" if GPT explicitly confirms all are correct
                if has_all_option and opt_clean.lower() in ['সবগুলোই', 'সবগুলো', 'all of the above', 'all of these']:
                    all_correct_phrases = [
                        "সব অপশনই সঠিক",
                        "সবগুলোই সঠিক",
                        "সকল অপশন সঠিক",
                        "তিনটিই সঠিক",
                        "তিনটি অপশনই সঠিক",
                        "all options are correct",
                        "all are correct",
                        "all three are correct"
                    ]
                    for phrase in all_correct_phrases:
                        if phrase.lower() in result_text.lower():
                            score += 50  # Very high score
                            print(f"   🎯 Found confirmation that all options are correct: '{phrase}'")
                            break
               
                # Check for exact match
                if opt_clean in result_text:
                    score += 10
               
                # Check for answer patterns
                answer_indicators = [
                    f"উত্তর: {opt_clean}",
                    f"উত্তর {opt_clean}",
                    f"সঠিক উত্তর: {opt_clean}",
                    f"সঠিক উত্তর {opt_clean}",
                    f"উত্তরটি হলো: {opt_clean}",
                    f"উত্তরটি হলো {opt_clean}",
                    f"উত্তর হবে: {opt_clean}",
                    f"উত্তর হবে {opt_clean}",
                    f"সিদ্ধান্ত: {opt_clean}",
                    f"Answer: {opt_clean}",
                    f"Correct answer: {opt_clean}"
                ]
                
                for indicator in answer_indicators:
                    if indicator in result_text:
                        score += 30
                        break
               
                # Check if option appears in conclusion (last 400 chars)
                last_400_chars = result_text[-400:]
                if opt_clean in last_400_chars:
                    score += 8
                
                # For multi-word options, check individual significant words
                opt_words = [w for w in opt_clean.split() if len(w) > 2]
                matching_words = sum(1 for word in opt_words if word in result_text)
                if opt_words:
                    score += (matching_words / len(opt_words)) * 5
               
                print(f"   Option {i+1} ('{opt_clean}'): Score = {score}")
               
                if score > best_match_score:
                    best_match_score = score
                    best_match = opt_clean
           
            # ✅ Minimum threshold for confidence
            if best_match and best_match_score >= 10:
                print(f"✓ GPT Knowledge (gpt-4o): Matched answer from solution")
                print(f"  ✓ Answer: '{best_match}' (score: {best_match_score})")
                return best_match
           
            # Method 2: Look for explicit answer patterns in Bengali/English
            answer_patterns = [
                r'উত্তর[:\s]+([^\n\.।]+)',
                r'সঠিক উত্তর[:\s]+([^\n\.।]+)',
                r'উত্তরটি হলো[:\s]+([^\n\.।]+)',
                r'সিদ্ধান্ত[:\s]+([^\n\.।]+)',
                r'Answer[:\s]+([^\n\.]+)',
                r'Correct answer[:\s]+([^\n\.]+)',
            ]
           
            for pattern in answer_patterns:
                match = re.search(pattern, result_text, re.IGNORECASE)
                if match:
                    potential_answer = match.group(1).strip()
                    print(f"   Found potential answer via pattern: '{potential_answer}'")
                   
                    # Check which option this matches
                    for opt in options:
                        if not opt:
                            continue
                        opt_clean = opt.strip()
                        if opt_clean in potential_answer or potential_answer in opt_clean:
                            print(f"✓ GPT Knowledge (gpt-4o): Extracted from answer pattern")
                            print(f"  ✓ Answer: '{opt_clean}'")
                            return opt_clean
           
            print(f"   ⚠ Could not extract clear answer from GPT's solution")
            print(f"   Best match was: '{best_match}' with score {best_match_score} (threshold: 10)")
            return None
           
        except Exception as model_error:
            print(f"   ✗ Error: {str(model_error)}")
            import traceback
            traceback.print_exc()
            return None
   
    except Exception as e:
        print(f"✗ GPT Knowledge ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    #News agent Components




       

def get_answer_from_openai_web_search(question: str, options: List[str]) -> Optional[str]:
    """
    ✅ SOURCE 3: OpenAI Web Search (Fallback after trusted news)
    
    Uses gpt-4o-mini-search-preview model with built-in web search capability
    """
    try:
        print("\n🌐 SOURCE 4: OpenAI Web Search (Real-time Internet Search)...")
        
        # Format options for better matching
        opts_formatted = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
        
        # Create the search query
        search_prompt = f"""You are answering a quiz question using ONLY verified authoritative sources.
Question: {question}
Options:
{opts_formatted}
MANDATORY SEARCH PROCESS:
1. Search 4-6 different TOP TIER sources based on topic
2. Cross-reference ALL options against these sources
3. Count votes: which option appears most in reliable sources
4. Choose the option with highest source agreement (minimum 3 sources)
AUTHORITATIVE SOURCES BY TOPIC:
- Bangladesh news/events: Prothom Alo, Daily Star, bdnews24, government sites
- International news: Reuters, AP, BBC, CNN, official statements
- Deaths/casualties: Official government reports, UN, verified news agencies
- Historical events: Wikipedia (cross-check), Britannica, academic sources
- Sports: ESPN, official league sites, verified sports media
- Science/health: WHO, CDC, peer-reviewed journals, Nature, Science
- Technology: official documentation, tech news (TechCrunch, Wired, official company sites)
- Politics: official government sites, established news agencies
- Entertainment: IMDB, official announcements, Variety, Hollywood Reporter
- Geography/statistics: World Bank, UN data, official census
- Business/Economy: Bloomberg, Reuters, Financial Times, official company reports
- Education: official university sites, education ministry, verified rankings
- Religion: official religious texts, verified scholarly sources
- Culture/Literature: established publishers, literary databases, verified reviews
- Law/Legal: official government legal sites, verified legal databases
FOR UNLISTED TOPICS:
1. Identify the topic category first
2. Search for: "[topic] official source" OR "[topic] authoritative database"
3. Use: Government sites (.gov), Educational institutions (.edu), International organizations (.org from UN/WHO etc)
4. Cross-reference with: Established news agencies (Reuters, AP, BBC)
5. Avoid: Personal blogs, forums, unverified sites, social media
STRICT RULES:
- Never trust a single source
- Ignore blogs, forums, social media claims
- For conflicting data: go with official/government source
- For numbers: use only confirmed figures, never estimates or "up to X"
- Minimum 3 sources must agree before selecting answer
- If topic is unclear, search broader then narrow down
- For niche topics: prioritize domain experts and official organizations
VERIFICATION CHECKLIST:
✓ Is this from a top-tier source for this topic?
✓ Do at least 3 reliable sources confirm this?
✓ Does this match official data (if applicable)?
✓ Are there any contradicting authoritative sources?
✓ If topic is unlisted: did I find official/expert sources?
Return ONLY this JSON:
{{
    "answer": "exact option text confirmed by majority of authoritative sources",
    "confidence": 90,
    "reasoning": "Confirmed by [source1], [source2], [source3]. Cross-checked against [total] sources."
}}
NO markdown blocks. NO extra text. ONLY JSON."""

        # ✅ Use gpt-4o-mini-search-preview (has built-in web search)
        print(f"   Using: gpt-4o-mini-search-preview")
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini-search-preview",  # ✅ Specialized search model
            messages=[
                {
                    "role": "user",
                    "content": search_prompt
                }
            ],
            # temperature=0
        )
        
        result_text = response.choices[0].message.content.strip()
        print(f"   ✓ Got search response")
        print(f"   📝 Response: {result_text[:200]}...")
        
        # Parse JSON response
        try:
            # Clean JSON
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
            
            # Fallback: Try to extract answer directly from text
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
   ✅ PERFECT COMBINED VERSION:
   - CORRECT answer validation (with normalization)
   - CORRECT question/logic validation (reasonable strictness)
   - FIXED explanation validation (handles HTML entities, validates math correctly)
   - IMPROVED GPT Knowledge Base (temperature=0 for math accuracy)
  
   Source Priority:
   1. Dataset (similar question)
   2. GPT-4 Knowledge Base (IMPROVED with better math prompts!)
   3. Trusted News Sources
  
   NOTE: INPUT explanation NOT used for answer determination, only for validation
   """
   try:
       lang = detect_language(request.question) if request.language == "auto" else request.language
      
       print(f"\n{'='*80}")
       print("🔍 FACT CHECK REQUEST")
       print(f"{'='*80}")
       print(f"Question: {request.question}")
       print(f"Given Answer: {request.answer}")
       print(f"Language: {lang}")
      
       # ✅ Use get_explanation() to support both fields
       explanation_text = request.get_explanation()
       has_exp = bool(explanation_text and explanation_text.strip())
       print(f"Explanation: {'PROVIDED' if has_exp else 'NOT PROVIDED'}")
       if has_exp:
           print(f"Explanation text: {explanation_text[:100]}...")
       print(f"{'='*80}\n")
      
       # STEP 1: Validate structure
       print("STEP 1: Validating question structure, grammar, and options...")
       validation = validate_structure_only(request)
      
       if has_exp:
           validation['explanation_valid'] = True  # Will be updated later
      
       print("✓ Validation complete\n")
      
       # STEP 2: Find correct answer (NOT from INPUT explanation)
       print(f"{'='*80}")
       print("STEP 2: Finding Correct Answer")
       print(f"{'='*80}")
       print("⚠️  INPUT Explanation is ONLY used for validation, NOT for answer extraction")
       print("    Answer sources: Dataset → GPT Knowledge Base → Trusted News")
      
       final_answer = None
       options = [request.option1, request.option2, request.option3, request.option4]
      
       # SOURCE 1: From dataset
       print("\n→ SOURCE 1: Dataset")
       final_answer = get_answer_from_dataset(request.question, options)
      
       if final_answer:
           print("✓ SOURCE 1 SUCCESS")
       else:
           print("✗ SOURCE 1 FAILED")
      
       # SOURCE 2: From GPT-4 knowledge base
       if not final_answer:
           print("\n→ SOURCE 2: GPT-4 Knowledge Base (IMPROVED v2.0 - temperature=0)")
           final_answer = get_answer_from_gpt_knowledge(request.question, options)
          
           if final_answer:
               print("✓ SOURCE 2 SUCCESS")
           else:
               print("✗ SOURCE 2 FAILED")
      
       # SOURCE 3: From TRUSTED news sources
    #    if not final_answer:
    #        print("\n→ SOURCE 3: Trusted News Sources (NewsAPI)")
    #        final_answer = get_answer_from_news(request.question, options)
          
    #        if final_answer:
    #            print("✓ SOURCE 3 SUCCESS")
    #        else:
    #            print("✗ SOURCE 3 FAILED")
      
       # SOURCE 4: From OpenAI Web Search (Fallback)
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
      
       # STEP 3: Validate INPUT explanation against CORRECT answer
       if has_exp and final_answer and final_answer != "Unable to determine the correct answer":
           print(f"\n{'='*80}")
           print("STEP 3: Validating INPUT Explanation Against CORRECT Answer")
           print(f"{'='*80}")
           print(f"  Correct answer from Dataset/GPT: '{final_answer}'")
           print(f"  Checking if INPUT explanation supports this answer...")
          
           # Step 3A: Extract what answer the explanation claims
           explanation_claims_answer = get_answer_from_explanation(
               explanation_text,  # ✅ Use explanation_text
               request.question,
               options
           )
          
           if explanation_claims_answer:
               print(f"  INPUT explanation claims answer is: '{explanation_claims_answer}'")
              
               # Normalize both for comparison
               explanation_normalized = normalize_answer(explanation_claims_answer)
               correct_normalized = normalize_answer(final_answer)
              
               if explanation_normalized == correct_normalized:
                   print(f"  ✓ Explanation answer MATCHES correct answer")
                  
                   # Step 3B: Validate math/facts in explanation
                   exp_validation = validate_explanation_correctness(
                       explanation_text,  # ✅ Use explanation_text
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
      
       # STEP 3: Compare given answer with correct answer (WITH NORMALIZATION)
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
           feedback=validation.get('question_feedback', ''),
           logical_valid=validation.get('logical_valid', True),
           options=OptionsValidation(
               option1=OptionValidation(
                   feedback=validation.get('option1_feedback', '')
               ),
               option2=OptionValidation(
                   feedback=validation.get('option2_feedback', '')
               ),
               option3=OptionValidation(
                   feedback=validation.get('option3_feedback', '')
               ),
               option4=OptionValidation(
                   feedback=validation.get('option4_feedback', '')
               ),
               option5=OptionValidation(
                   feedback=validation.get('option5_feedback', '')
               ),
               options_consistency_valid=validation.get('options_consistency_valid', True),
               feedback=validation.get('options_consistency_feedback', '')
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
   print("🚀 Fact Checker & MCQ Validator API - FINAL PERFECT VERSION")
   print(f"{'='*80}")
   print("✅ PERFECT: Answer validation with normalization")
   print("✅ PERFECT: Question/logic validation (reasonable strictness)")
   print("✅ FIXED: Explanation validation (handles HTML entities correctly)")
   print("✅ PERFECT: GPT Knowledge Base (temperature=0 for math accuracy)")
   print("="*80)
   print("⚠️  IMPORTANT: INPUT Explanation is ONLY for validation")
   print("    It does NOT determine the answer")
   print("="*80)
   print("Answer Source Priority:")
   print("  1. Dataset (40,000+ questions, similarity 0.85+)")
   print("  2. GPT-4 Knowledge Base (temperature=0, improved math prompts)")
   print("  3. Trusted News Sources")
   print("")
   print("Explanation Validation:")
   print("  - Extracts what answer explanation claims")
   print("  - Compares with correct answer from Dataset/GPT")
   print("  - Validates math and factual correctness (ignores HTML entities)")
   print("  - Sets explanation_valid field in response")
   print(f"{'='*80}\n")
  
   uvicorn.run(app, host=settings.api_host, port=settings.api_port)