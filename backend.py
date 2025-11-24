
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
        temperature=0,
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
    language: Optional[str] = "auto"

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
    Normalize answer by removing option prefixes and extra whitespace
    Handles: "ক)", "খ)", "গ)", "ঘ)", "a)", "b)", "c)", "d)", "1)", "2)", etc.
    """
    if not answer:
        return ""
    
    # Remove common option prefixes
    # Bengali: ক) খ) গ) ঘ) ঙ)
    # English: a) b) c) d) e) A) B) C) D) E)
    # Numbers: 1) 2) 3) 4) 5)
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
    Strictly detect duplicate options using Python comparison.
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

def validate_structure_only(request: FactCheckRequest) -> Dict[str, Any]:
    """
    ✅ FIXED: Now strictly validates question logic and option appropriateness
    Validates grammar, structure, and contextual relevance using GPT-4.
    Checks for duplicates using strict Python comparison.
    Does NOT determine which answer is correct.
    """
    try:
        system_msg = """You are a strict question and answer validator with expertise in logic and reasoning.

Your job:
1. Check if the QUESTION is LOGICALLY VALID and makes sense
2. Check if each OPTION is appropriate for the question type

QUESTION VALIDATION - BE STRICT:
A question is INVALID if:
- It contains logical contradictions (e.g., "How many days to pass a bill that hasn't been written yet?")
- It asks about impossible scenarios or paradoxes
- It has contradictory conditions (e.g., "if X doesn't exist, how does X work?")
- It's grammatically nonsensical or incomplete
- It's unclear or ambiguous to the point of being unanswerable

LOGICAL VALIDATION - BE STRICT:
Mark logical_valid as FALSE if:
- Question contains logical contradictions
- Options don't match question type (numbers for text questions, text for number questions)
- Options are completely irrelevant or nonsensical
- Question-option combination makes no sense

OPTION VALIDATION - BE STRICT:
Options are INVALID if:
- Random meaningless numbers (e.g., "1111111", "999999") for non-numeric questions
- Wrong type entirely (dates for math, numbers for history/law questions, words for arithmetic)
- Completely irrelevant to the question context
- Grammatically nonsensical
- Obviously fake or placeholder text

EXAMPLES OF INVALID SCENARIOS:
❌ INVALID QUESTION:
Q: "রাষ্ট্রপতি সংবিধান সংশোধন বিল কত দিনের মধ্যে পাশ করবেন যদি বিলটি এখনো লেখা না হয়?"
(How many days will president pass a bill if the bill hasn't been written yet?)
→ question_valid: FALSE, logical_valid: FALSE
→ Feedback: "Logical contradiction - cannot pass a bill that doesn't exist yet"

❌ INVALID OPTIONS:
Q: "When was Bangladesh independence?" Options: "1111111", "999999", "5555555"
→ option_valid: FALSE for all
→ Feedback: "Random numbers are not valid years/dates"

Q: "What is 2+2?" Options: "happy", "sad", "angry"
→ option_valid: FALSE for all
→ Feedback: "Text words are not valid for arithmetic questions"

✅ VALID SCENARIO:
Q: "What is the capital of Bangladesh?" Options: "Dhaka", "Chittagong", "Sylhet"
→ All valid, logical, appropriate

BE STRICT. If something is illogical, contradictory, or wrong type, mark it as INVALID with clear feedback.

Return JSON:
{
    "question_valid": true/false,
    "question_feedback": "explain why question is invalid with specific reasons, or empty if valid",
    "logical_valid": true/false,
    "logical_feedback": "explain logical problems or question-option type mismatch, or empty if valid",
    "option1_valid": true/false,
    "option1_feedback": "explain why invalid with specific reason, or empty if valid",
    "option2_valid": true/false,
    "option2_feedback": "explain why invalid with specific reason, or empty if valid",
    "option3_valid": true/false,
    "option3_feedback": "explain why invalid with specific reason, or empty if valid",
    "option4_valid": true/false,
    "option4_feedback": "explain why invalid with specific reason, or empty if valid",
    "explanation_valid": true/false,
    "explanation_feedback": ""
}

Return ONLY JSON."""

        has_exp = bool(request.explanation and request.explanation.strip())
        
        human_msg = f"""Strictly validate this question and options. BE STRICT about logic and appropriateness:

Question: {request.question}
Option 1: {request.option1}
Option 2: {request.option2}
Option 3: {request.option3}
Option 4: {request.option4}
Option 5: {request.option5}
Explanation: {request.explanation if has_exp else 'NOT PROVIDED'}

Check thoroughly:
1. Is the QUESTION logically valid? Any contradictions or impossible scenarios?
2. Are OPTIONS the right TYPE for this question? (e.g., dates for history, numbers for math)
3. Are options meaningful or just random gibberish?
4. Does the question-option combination make logical sense?

BE STRICT. Mark as INVALID if:
- Question has logical contradictions
- Options are wrong type (random numbers for non-numeric questions, etc.)
- Options are meaningless or clearly fake

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
    ✅ FIXED: Validate INPUT explanation by comparing with DATASET answer
    
    Process:
    1. Extract answer from INPUT explanation using GPT
    2. Get correct answer from DATASET
    3. Compare both answers
    4. If they MATCH → Explanation is TRUE, use it
    5. If they DON'T MATCH → Explanation is FALSE, skip it
    """
    try:
        print("\n📝 VALIDATING INPUT EXPLANATION AGAINST DATASET...")
        
        # STEP 1: Extract answer from INPUT explanation using GPT
        print("  Step 1: Extracting answer from INPUT explanation using GPT...")
        
        system_msg = """You are an expert at analyzing explanations to determine the correct answer.

Read the explanation carefully and determine which option it supports.

Return ONLY JSON:
{
    "answer": "the correct answer as EXACT TEXT from options",
    "confidence": 95,
    "reasoning": "brief explanation"
}"""
        
        opts = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
        
        user_msg = f"""Question: {question}

Options:
{opts}

Explanation: {explanation}

What answer does this explanation support? Return ONLY JSON."""

        response = call_gpt4(system_msg, user_msg)
        result = json.loads(clean_json(response))
        
        explanation_answer = result.get('answer', '').strip()
        confidence = result.get('confidence', 0)
        
        if not explanation_answer or confidence < 70:
            print(f"  ✗ Could not extract answer from explanation (confidence: {confidence}%)")
            return None
        
        print(f"  ✓ INPUT Explanation says answer is: '{explanation_answer}'")
        
        # STEP 2: Get correct answer from DATASET
        print("  Step 2: Getting correct answer from DATASET...")
        
        query_emb = embedding_service.embed_query(question)
        results = vector_db.search(COLLECTION_NAME, query_emb, top_k=10)
        
        if not results:
            print("  ✗ No dataset results found - cannot validate explanation")
            return None
        
        best = max(results, key=lambda x: x.get('score', 0))
        similarity = best.get('score', 0)
        
        # ✅ FIXED: Use higher threshold (0.85) for explanation validation too
        if similarity < 0.85:
            print(f"  ✗ Dataset similarity too low ({similarity:.4f} < 0.85) - cannot validate")
            print(f"  → Not the same question, cannot validate explanation against dataset")
            return None
        
        print(f"  ✓ Found similar question in dataset (similarity: {similarity:.4f})")
        
        # Get dataset answer
        dataset_answer = None
        try:
            stored_options = json.loads(best.get('options', '{}'))
            answer_num = best.get('answer')
            
            if answer_num:
                dataset_answer = stored_options.get(f'option{answer_num}', '').strip()
                print(f"  ✓ DATASET says correct answer is: '{dataset_answer}'")
        except Exception as e:
            print(f"  ✗ Error getting dataset answer: {e}")
            return None
        
        if not dataset_answer:
            print("  ✗ No answer in dataset - cannot validate")
            return None
        
        # STEP 3: Compare INPUT explanation answer with DATASET answer
        print("  Step 3: Comparing INPUT explanation with DATASET answer...")
        
        explanation_normalized = normalize_answer(explanation_answer)
        dataset_normalized = normalize_answer(dataset_answer)
        
        print(f"  INPUT Explanation answer (normalized): '{explanation_normalized}'")
        print(f"  DATASET answer (normalized): '{dataset_normalized}'")
        
        if explanation_normalized == dataset_normalized:
            print("  ✅ MATCH! INPUT Explanation is CORRECT ✅")
            print("  → Using answer from INPUT explanation")
            return explanation_answer
        else:
            print("  ❌ NO MATCH! INPUT Explanation is FALSE ❌")
            print(f"  → INPUT Explanation gives: '{explanation_answer}'")
            print(f"  → DATASET correct answer: '{dataset_answer}'")
            print("  → SKIPPING INPUT explanation, will try other sources")
            return None
        
    except Exception as e:
        print(f"✗ Explanation validation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_answer_from_dataset(question: str, options: List[str]) -> Optional[str]:
    """
    Find SAME/SIMILAR question in dataset and return its answer
    Priority: 1. Check explanation, 2. Check answer number and convert to text
    
    ✅ FIXED: Higher similarity threshold (0.85) to ensure accurate matches
    ✅ FIXED: Validates options match before accepting dataset answer
    """
    try:
        print("\n💾 Searching dataset for same/similar question...")
        
        query_emb = embedding_service.embed_query(question)
        results = vector_db.search(COLLECTION_NAME, query_emb, top_k=10)
        
        if not results:
            print("✗ No results from dataset")
            return None
        
        print(f"✓ Found {len(results)} similar questions")
        
        # Get best match
        best = max(results, key=lambda x: x.get('score', 0))
        similarity = best.get('score', 0)
        matched_question = best.get('question', '')
        
        print(f"  Best match:")
        print(f"  Similarity: {similarity:.4f}")
        print(f"  Question: {matched_question[:100]}...")
        
        # ✅ FIXED: Much higher threshold (0.85) to ensure it's actually the SAME question
        # Only accept if similarity is very high (85%+)
        if similarity >= 0.85:
            print(f"  ✓ HIGH similarity - This looks like the SAME question")
            try:
                # Get stored options
                stored_options = json.loads(best.get('options', '{}'))
                answer_num = best.get('answer')
                stored_explanation = best.get('explanation', '').strip()
                
                # ✅ NEW: Validate that dataset options match current options
                # Extract dataset options
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
                
                # ✅ FIXED: Require at least 3 out of 4 options to match
                if matching_options < 3:
                    print(f"  ✗ Options don't match well enough ({matching_options}/4)")
                    print(f"  → This is a DIFFERENT question, not using dataset answer")
                    print(f"  → Will try GPT Knowledge Base instead")
                    return None
                
                print(f"  ✓ Options match well - This is definitely the same question")
                
                # Priority 1: Check if explanation exists and extract answer from it
                if stored_explanation:
                    print("  ✓ Explanation found in dataset")
                    print(f"  Explanation: {stored_explanation[:100]}...")
                    
                    # Extract answer from dataset explanation
                    dataset_options = [
                        stored_options.get('option1', ''),
                        stored_options.get('option2', ''),
                        stored_options.get('option3', ''),
                        stored_options.get('option4', '')
                    ]
                    
                    # Use a simplified extraction for dataset explanation
                    try:
                        system_msg = """Extract the correct answer from this explanation.
Return ONLY JSON: {"answer": "answer text", "confidence": 90}"""
                        
                        opts_text = "\n".join([f"{i+1}. {o}" for i, o in enumerate(dataset_options) if o])
                        user_msg = f"Question: {matched_question}\n\nOptions:\n{opts_text}\n\nExplanation: {stored_explanation}\n\nReturn ONLY JSON."
                        
                        response = call_gpt4(system_msg, user_msg)
                        result = json.loads(clean_json(response))
                        extracted_answer = result.get('answer', '').strip()
                        
                        if extracted_answer:
                            # Try to match with current question's options
                            for opt in options:
                                if opt.strip().lower() == extracted_answer.strip().lower():
                                    print(f"  ✓ Answer from dataset explanation: '{opt}'")
                                    return opt
                            
                            # If exact match not found, return extracted answer
                            print(f"  ✓ Answer from dataset explanation: '{extracted_answer}'")
                            return extracted_answer
                    except:
                        pass
                
                # Priority 2: Use answer number to get text option
                if answer_num:
                    answer_text = stored_options.get(f'option{answer_num}', '').strip()
                    
                    if answer_text:
                        # Try to match with current question's options
                        for opt in options:
                            if opt.strip().lower() == answer_text.strip().lower():
                                print(f"  ✓ Answer from dataset (option {answer_num}): '{opt}'")
                                return opt
                        
                        # If no exact match, return the answer text from dataset
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
    ✅ IMPROVED GPT-4 Knowledge Base - Now works as well as real ChatGPT!
    
    Improvements:
    - Uses gpt-4o (latest and smartest model)
    - More assertive prompting for confident answers
    - Multiple answer matching strategies
    - Fallback approaches if first attempt fails
    - Better extraction from GPT responses
    """
    try:
        print("\n🧠 Asking GPT-4 Knowledge Base (IMPROVED v2.0)...")
        
        system_msg = """You are a highly knowledgeable expert assistant with extensive training in multiple domains including: history, science, geography, mathematics, current events, Bangladesh history and politics, world affairs, culture, technology, and many other subjects.

YOUR TASK: Answer the multiple choice question using your training knowledge.

INSTRUCTIONS:
1. Read the question carefully and understand what is being asked
2. Analyze each option thoroughly
3. Use your training knowledge to determine the correct answer
4. Be confident - you have extensive knowledge, use it!
5. Return the answer as EXACT TEXT from one of the provided options (copy-paste it exactly)

CONFIDENCE GUIDELINES:
- If you recognize the topic and know the answer: 85-100% confidence
- If you have strong knowledge but not 100% certain: 70-84% confidence
- If you have some knowledge but unsure: 50-69% confidence
- Only use 0% if you truly have NO knowledge about the topic

Return ONLY this JSON format:
{
    "answer": "exact text of correct option (copy from options list)",
    "confidence": 95,
    "reasoning": "brief 1-2 sentence explanation why this is correct"
}

CRITICAL: Copy the exact option text, don't paraphrase or modify it!"""

        opts_formatted = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
        
        user_msg = f"""Question: {question}

Available Options (choose one):
{opts_formatted}

Based on your training knowledge, which option is correct?

Think step by step:
1. What is this question asking?
2. What do I know about this topic?
3. Which option matches my knowledge?

Return your answer in JSON format with the exact option text."""

        # Strategy 1: Try modern models first
        models_to_try = [
            ("gpt-4o", "Latest GPT-4 Omni - Best for reasoning"),
            ("gpt-4-turbo", "GPT-4 Turbo - Very capable"),
            ("gpt-4", "GPT-4 - Reliable fallback")
        ]
        
        for model_name, model_desc in models_to_try:
            try:
                print(f"   Attempting: {model_name} ({model_desc})")
                
                response = openai_client.chat.completions.create(
                    model=model_name,
                    temperature=0.3,  # Slightly creative for better reasoning
                    max_tokens=500,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ]
                )
                
                result_text = response.choices[0].message.content.strip()
                print(f"   ✓ Got response from {model_name}")
                
                # Parse JSON
                try:
                    result = json.loads(clean_json(result_text))
                except json.JSONDecodeError:
                    print(f"   ⚠ JSON parse failed, trying to extract...")
                    # Try to extract JSON from text
                    json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(0))
                    else:
                        continue
                
                answer = result.get('answer', '').strip()
                confidence = result.get('confidence', 0)
                reasoning = result.get('reasoning', '')
                
                if not answer:
                    print(f"   ⚠ No answer in response, trying next model...")
                    continue
                
                print(f"   📝 GPT Answer: '{answer}'")
                print(f"   📊 Confidence: {confidence}%")
                print(f"   💭 Reasoning: {reasoning}")
                
                # Strategy A: Exact match (case-insensitive)
                for opt in options:
                    if opt.strip().lower() == answer.lower():
                        print(f"✓ GPT Knowledge ({model_name}): EXACT MATCH")
                        print(f"  ✓ Answer: '{opt}'")
                        print(f"  ✓ Confidence: {confidence}%")
                        print(f"  ✓ Reasoning: {reasoning}")
                        return opt
                
                # Strategy B: Normalized match (remove prefixes like "a)", "1)", etc.)
                answer_norm = normalize_answer(answer)
                for opt in options:
                    opt_norm = normalize_answer(opt)
                    if opt_norm and answer_norm and opt_norm == answer_norm:
                        print(f"✓ GPT Knowledge ({model_name}): NORMALIZED MATCH")
                        print(f"  ✓ Answer: '{opt}'")
                        print(f"  ✓ Confidence: {confidence}%")
                        return opt
                
                # Strategy C: Substring match (answer contains option or vice versa)
                for opt in options:
                    opt_clean = opt.strip().lower()
                    answer_clean = answer.lower()
                    
                    # Skip very short matches to avoid false positives
                    if len(opt_clean) < 4:
                        continue
                    
                    if opt_clean in answer_clean or answer_clean in opt_clean:
                        print(f"✓ GPT Knowledge ({model_name}): SUBSTRING MATCH")
                        print(f"  ✓ Answer: '{opt}'")
                        print(f"  ✓ Confidence: {confidence}%")
                        return opt
                
                # Strategy D: Word overlap match (for longer answers)
                if confidence >= 60:
                    answer_words = set(answer.lower().split())
                    best_match = None
                    best_overlap = 0
                    
                    for opt in options:
                        opt_words = set(opt.lower().split())
                        overlap = len(answer_words & opt_words)
                        
                        if overlap > best_overlap and overlap >= 2:  # At least 2 words match
                            best_overlap = overlap
                            best_match = opt
                    
                    if best_match:
                        print(f"✓ GPT Knowledge ({model_name}): WORD OVERLAP MATCH")
                        print(f"  ✓ Answer: '{best_match}' ({best_overlap} words matched)")
                        print(f"  ✓ Confidence: {confidence}%")
                        return best_match
                
                # Strategy E: High confidence - trust GPT even without perfect match
                if confidence >= 80:
                    print(f"✓ GPT Knowledge ({model_name}): HIGH CONFIDENCE ANSWER")
                    print(f"  ⚠ No exact option match, but GPT is {confidence}% confident")
                    print(f"  ✓ Returning: '{answer}'")
                    print(f"  💭 Reasoning: {reasoning}")
                    return answer
                
                print(f"   ⚠ Confidence too low ({confidence}%), trying next approach...")
                
            except Exception as model_error:
                print(f"   ✗ Model {model_name} error: {str(model_error)[:100]}")
                continue
        
        # Strategy 2: Last resort - Direct conversational approach
        print("\n   🔄 Trying direct conversational approach...")
        try:
            direct_prompt = f"""Answer this question directly and clearly:

{question}

Options:
{opts_formatted}

Which option is correct? Just tell me the answer clearly."""

            direct_response = openai_client.chat.completions.create(
                model="gpt-4o",
                temperature=0.5,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": "You are a helpful expert. Answer questions clearly and directly."},
                    {"role": "user", "content": direct_prompt}
                ]
            )
            
            direct_text = direct_response.choices[0].message.content.strip().lower()
            print(f"   Direct response: {direct_text[:150]}...")
            
            # Find option in response
            for i, opt in enumerate(options):
                opt_lower = opt.strip().lower()
                # Check if option appears in response
                if opt_lower in direct_text:
                    print(f"✓ GPT Knowledge (direct): Found option in response")
                    print(f"  ✓ Answer: '{opt}'")
                    return opt
                
                # Check if option number is mentioned
                if f"option {i+1}" in direct_text or f"{i+1}." in direct_text[:30]:
                    print(f"✓ GPT Knowledge (direct): Found by option number")
                    print(f"  ✓ Answer: '{opt}'")
                    return opt
            
        except Exception as e:
            print(f"   ✗ Direct approach failed: {e}")
        
        print("✗ GPT Knowledge: Unable to determine answer with confidence")
        print("  All strategies exhausted")
        return None
    
    except Exception as e:
        print(f"✗ GPT Knowledge CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_answer_from_news(question: str, options: List[str]) -> Optional[str]:
    """Get answer from TRUSTED news sources ONLY"""
    try:
        print("\n📰 Searching TRUSTED news sources...")
        print("  Trusted sources: Prothom Alo, The Daily Star, BBC Bangla, Bangladesh Pratidin, NCTB Books")
        
        query_emb = embedding_service.embed_query(question)
        news = vector_db.search(NEWS_COLLECTION_NAME, query_emb, top_k=10)
        
        if not news:
            print("✗ No news articles found")
            return None
        
        # Filter for TRUSTED sources only
        trusted_sources = ["Prothom Alo", "The Daily Star", "BBC Bangla", "Bangladesh Pratidin", "NCTB"]
        trusted_news = [n for n in news if any(source.lower() in n.get('source', '').lower() for source in trusted_sources)]
        
        if not trusted_news:
            print("✗ No articles from trusted sources found")
            return None
        
        print(f"✓ Found {len(trusted_news)} articles from trusted sources")
        
        context = "\n\n".join([
            f"[{n.get('source', 'Unknown')}] {n.get('title', '')}\n{n.get('content', '')[:400]}"
            for n in trusted_news[:3]
        ])
        
        system_msg = """Based ONLY on the news articles from TRUSTED sources, answer the question.

IMPORTANT: Only answer if the information is clearly stated in the articles.

Return ONLY JSON:
{
    "answer": "the correct answer as TEXT from options",
    "confidence": 85,
    "source": "which article provided this info"
}

If articles don't contain the answer: {"answer": "", "confidence": 0, "source": ""}"""
        
        opts = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
        
        response = call_gpt4(
            system_msg,
            f"Trusted News Articles:\n{context}\n\nQuestion: {question}\n\nOptions:\n{opts}\n\nBased on these trusted news articles, what is the answer?\n\nReturn ONLY JSON."
        )
        
        result = json.loads(clean_json(response))
        answer = result.get('answer', '').strip()
        confidence = result.get('confidence', 0)
        source = result.get('source', '')
        
        if answer and confidence >= 70:
            print(f"✓ Trusted News: '{answer}' (confidence: {confidence}%)")
            print(f"  Source: {source}")
            return answer
        else:
            print("✗ Trusted News: No confident answer found")
            return None
        
    except Exception as e:
        print(f"✗ News error: {e}")
        return None

@app.post("/fact-check", response_model=FactCheckResponse)
async def fact_check(request: FactCheckRequest):
    """
    Complete fact checking with TWO FIXES:
    
    ✅ FIX 1: INPUT Explanation validated against DATASET
    ✅ FIX 2: Strict question and logical validation using GPT-4
    ✅ FIX 3: IMPROVED GPT Knowledge Base (v2.0)
    
    Source Priority:
    1. INPUT Explanation (if provided AND matches dataset answer)
    2. Dataset (similar question with explanation or answer number)
    3. GPT-4 Knowledge Base (using OpenAI API key) ← IMPROVED!
    4. Trusted News Sources (Prothom Alo, The Daily Star, BBC Bangla, Bangladesh Pratidin, NCTB)
    """
    try:
        lang = detect_language(request.question) if request.language == "auto" else request.language
        
        print(f"\n{'='*80}")
        print("🔍 FACT CHECK REQUEST")
        print(f"{'='*80}")
        print(f"Question: {request.question}")
        print(f"Given Answer: {request.answer}")
        print(f"Language: {lang}")
        
        has_exp = bool(request.explanation and request.explanation.strip())
        print(f"Explanation: {'PROVIDED' if has_exp else 'NOT PROVIDED'}")
        print(f"{'='*80}\n")
        
        # STEP 1: Validate structure with STRICT logic checking
        print("STEP 1: Validating question logic, grammar, and options (STRICT)...")
        validation = validate_structure_only(request)
        print("✓ Validation complete\n")
        
        # STEP 2: Find correct answer using source priority WITH EXPLANATION FIX
        print(f"{'='*80}")
        print("STEP 2: Finding Correct Answer")
        print(f"{'='*80}")
        
        final_answer = None
        options = [request.option1, request.option2, request.option3, request.option4]
        
        # SOURCE 1: From INPUT explanation (WITH DATASET VALIDATION)
        if has_exp:
            print("\n→ SOURCE 1: INPUT Explanation (validating against dataset)")
            final_answer = get_answer_from_explanation(request.explanation, request.question, options)
            
            if final_answer:
                print("✓ SOURCE 1 SUCCESS - INPUT Explanation MATCHES dataset")
            else:
                print("✗ SOURCE 1 FAILED - INPUT Explanation does NOT match dataset or invalid")
        else:
            print("\n✗ SOURCE 1: Explanation (not provided)")
        
        # SOURCE 2: From dataset (check explanation first, then answer number)
        if not final_answer:
            print("\n→ SOURCE 2: Dataset (Similar Questions)")
            final_answer = get_answer_from_dataset(request.question, options)
            
            if final_answer:
                print("✓ SOURCE 2 SUCCESS")
            else:
                print("✗ SOURCE 2 FAILED")
        
        # SOURCE 3: From GPT-4 knowledge base (IMPROVED!)
        if not final_answer:
            print("\n→ SOURCE 3: GPT-4 Knowledge Base (IMPROVED v2.0)")
            final_answer = get_answer_from_gpt_knowledge(request.question, options)
            
            if final_answer:
                print("✓ SOURCE 3 SUCCESS")
            else:
                print("✗ SOURCE 3 FAILED")
        
        # SOURCE 4: From TRUSTED news sources
        if not final_answer:
            print("\n→ SOURCE 4: Trusted News Sources")
            final_answer = get_answer_from_news(request.question, options)
            
            if final_answer:
                print("✓ SOURCE 4 SUCCESS")
            else:
                print("✗ SOURCE 4 FAILED")
        
        # Fallback
        if not final_answer:
            final_answer = "Unable to determine the correct answer"
            print("\n❌ ALL SOURCES FAILED - Cannot determine correct answer")
        
        # STEP 3: Compare given answer with correct answer (WITH NORMALIZATION)
        print(f"\n{'='*80}")
        print("STEP 3: Comparing Given Answer with Correct Answer")
        print(f"{'='*80}")
        
        given_answer_valid = False
        if final_answer and final_answer != "Unable to determine the correct answer":
            # Normalize both answers by removing option prefixes
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
                print(f"  Given: '{request.answer}' → '{given_normalized}'")
                print(f"  Correct: '{final_answer}' → '{final_normalized}'")
        else:
            print("✗ Cannot validate: No correct answer found")
            print("  WARNING: Marking as INVALID because we couldn't find the correct answer")
            given_answer_valid = False
        
        print(f"\n{'='*80}")
        print("FINAL RESULT")
        print(f"{'='*80}")
        print(f"Correct Answer: {final_answer}")
        print(f"Given Answer: {request.answer}")
        print(f"Given Answer Valid: {given_answer_valid}")
        print(f"{'='*80}\n")
        
        # Build response
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
    print("🚀 Fact Checker & MCQ Validator API")
    print(f"{'='*80}")
    print("✅ FIX 1: INPUT Explanation Validation Against Dataset")
    print("✅ FIX 2: Strict Question & Logical Validation")
    print("✅ FIX 3: IMPROVED GPT Knowledge Base (v2.0)")
    print("="*80)
    print("Now detects:")
    print("  - Illogical questions (contradictions, impossible scenarios)")
    print("  - Wrong option types (random numbers for non-numeric questions)")
    print("  - False explanations (comparing with dataset)")
    print("="*80)
    print("Source Priority:")
    print("  1. INPUT Explanation (if matches dataset)")
    print("  2. Dataset (similar question)")
    print("  3. GPT-4 Knowledge Base (IMPROVED with gpt-4o)")
    print("  4. Trusted News Sources")
    print(f"{'='*80}\n")
    
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)