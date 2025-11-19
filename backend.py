
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
    valid: bool
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
        r'^[ক-ঙ]\)\s*',  # Bengali options
        r'^[a-eA-E]\)\s*',  # English options
        r'^[1-5]\)\s*',  # Numbered options
        r'^[ক-ঙ]\s*।\s*',  # Bengali with vertical bar
        r'^[a-eA-E]\s*\.\s*',  # English with dot
        r'^[1-5]\s*\.\s*',  # Numbers with dot
    ]
    
    normalized = answer.strip()
    for pattern in patterns:
        normalized = re.sub(pattern, '', normalized)
    
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized.strip().lower()


def validate_structure_only(request: FactCheckRequest) -> Dict[str, Any]:
    """
    ONLY validates grammar and structure.
    Does NOT determine correct answer.
    """
    try:
        system_msg = """You are a grammar and structure validator.
Check ONLY:
1. Question: Is it grammatically correct and clear?
2. Options: Are they grammatically correct?
3. Explanation: If provided, is it grammatically correct?

DO NOT determine which answer is correct.
DO NOT check factual accuracy.
ONLY check grammar and clarity.

Return JSON:
{
    "question_valid": true/false,
    "question_feedback": "",
    "logical_valid": true/false,
    "logical_feedback": "",
    "option1_valid": true/false,
    "option1_feedback": "",
    "option2_valid": true/false,
    "option2_feedback": "",
    "option3_valid": true/false,
    "option3_feedback": "",
    "option4_valid": true/false,
    "option4_feedback": "",
    "options_consistency_valid": true/false,
    "options_consistency_feedback": "",
    "explanation_valid": true/false,
    "explanation_feedback": ""
}
Return ONLY JSON."""

        has_exp = bool(request.explanation and request.explanation.strip())
        
        human_msg = f"""Check grammar and structure only:
Question: {request.question}
Option 1: {request.option1}
Option 2: {request.option2}
Option 3: {request.option3}
Option 4: {request.option4}
Option 5: {request.option5}
Explanation: {request.explanation if has_exp else 'NOT PROVIDED'}

Return JSON."""

        response = call_gpt4(system_msg, human_msg)
        result = json.loads(clean_json(response))
        
        # Force explanation_valid = false if not provided
        if not has_exp:
            result['explanation_valid'] = False
            result['explanation_feedback'] = "Not provided"
        
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
    """Extract answer from valid explanation - IMPROVED VERSION"""
    try:
        print("\n📝 Extracting answer from explanation...")
        
        system_msg = """You are an expert at analyzing explanations to determine the correct answer.
Read the explanation carefully and determine which option it supports.

IMPORTANT: 
- Base your answer ONLY on what the explanation says
- The explanation will clearly indicate which option is correct
- Return the EXACT TEXT of the correct option from the provided options

Return ONLY JSON:
{
    "answer": "the correct answer as EXACT TEXT from options",
    "confidence": 95,
    "reasoning": "brief explanation of why this option matches the explanation"
}"""
        
        opts = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
        
        user_msg = f"""Question: {question}

Options:
{opts}

Explanation: {explanation}

Based on this explanation, what is the correct answer? Return ONLY JSON with the EXACT option text."""

        response = call_gpt4(system_msg, user_msg)
        result = json.loads(clean_json(response))
        
        answer = result.get('answer', '').strip()
        confidence = result.get('confidence', 0)
        reasoning = result.get('reasoning', '')
        
        if answer and confidence >= 70:
            print(f"✓ From explanation: '{answer}'")
            print(f"  Confidence: {confidence}%")
            print(f"  Reasoning: {reasoning}")
            return answer
        else:
            print(f"✗ Low confidence ({confidence}%) from explanation")
            return None
            
    except Exception as e:
        print(f"✗ Explanation extraction error: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_answer_from_dataset(question: str, options: List[str]) -> Optional[str]:
    """
    Find SAME/SIMILAR question in dataset and return its answer
    Priority: 1. Check explanation, 2. Check answer number and convert to text
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
        print(f"    Similarity: {similarity:.4f}")
        print(f"    Question: {matched_question[:100]}...")
        
        # Lower threshold for better matches
        if similarity >= 0.12:
            try:
                # Get stored options
                stored_options = json.loads(best.get('options', '{}'))
                answer_num = best.get('answer')
                stored_explanation = best.get('explanation', '').strip()
                
                # Priority 1: Check if explanation exists and extract answer from it
                if stored_explanation:
                    print("  ✓ Explanation found in dataset")
                    print(f"    Explanation: {stored_explanation[:100]}...")
                    
                    # Extract answer from dataset explanation
                    dataset_options = [
                        stored_options.get('option1', ''),
                        stored_options.get('option2', ''),
                        stored_options.get('option3', ''),
                        stored_options.get('option4', '')
                    ]
                    
                    extracted_answer = get_answer_from_explanation(
                        stored_explanation, 
                        matched_question, 
                        dataset_options
                    )
                    
                    if extracted_answer:
                        # Try to match with current question's options
                        for opt in options:
                            if opt.strip().lower() == extracted_answer.strip().lower():
                                print(f"    ✓ Answer from dataset explanation: '{opt}'")
                                return opt
                        
                        # If exact match not found, return extracted answer
                        print(f"    ✓ Answer from dataset explanation: '{extracted_answer}'")
                        return extracted_answer
                
                # Priority 2: Use answer number to get text option
                if answer_num:
                    answer_text = stored_options.get(f'option{answer_num}', '').strip()
                    
                    if answer_text:
                        # Try to match with current question's options
                        for opt in options:
                            if opt.strip().lower() == answer_text.strip().lower():
                                print(f"    ✓ Answer from dataset (option {answer_num}): '{opt}'")
                                return opt
                        
                        # If no exact match, return the answer text from dataset
                        print(f"    ✓ Answer from dataset (option {answer_num}): '{answer_text}'")
                        return answer_text
                    else:
                        print(f"    ✗ Could not get text for option {answer_num}")
                else:
                    print("    ✗ No answer number in dataset")
                    
            except Exception as e:
                print(f"    ✗ Error extracting from dataset: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"    ✗ Similarity too low ({similarity:.4f} < 0.12)")
        
        return None
        
    except Exception as e:
        print(f"✗ Dataset error: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_answer_from_gpt_knowledge(question: str, options: List[str]) -> Optional[str]:
    """Get answer from GPT-4 knowledge base using OpenAI API"""
    try:
        print("\n🧠 Asking GPT-4 Knowledge Base...")
        
        system_msg = """You are an expert with extensive knowledge. Answer this question based on your training data and knowledge.

IMPORTANT: Only answer if you are confident. If you don't know or are unsure, return confidence 0.

Return ONLY JSON:
{
    "answer": "the correct answer as TEXT from options (exact match)",
    "confidence": 90,
    "reasoning": "brief explanation of why this is correct"
}

If you don't know: {"answer": "", "confidence": 0, "reasoning": "Unknown"}"""
        
        opts = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
        
        response = call_gpt4(
            system_msg,
            f"Question: {question}\n\nOptions:\n{opts}\n\nBased on your knowledge, what is the correct answer?\n\nReturn ONLY JSON."
        )
        
        result = json.loads(clean_json(response))
        answer = result.get('answer', '').strip()
        confidence = result.get('confidence', 0)
        reasoning = result.get('reasoning', '')
        
        if answer and confidence >= 70:
            print(f"✓ GPT Knowledge: '{answer}'")
            print(f"  Confidence: {confidence}%")
            print(f"  Reasoning: {reasoning}")
            return answer
        else:
            print(f"✗ GPT Knowledge: Low confidence ({confidence}%) or no answer")
            return None
            
    except Exception as e:
        print(f"✗ GPT Knowledge error: {e}")
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
    Complete fact checking with UPDATED source priority:
    1. Explanation (if provided and valid)
    2. Dataset (similar question with explanation or answer number)
    3. GPT-4 Knowledge Base (using OpenAI API key)
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
        
        # STEP 1: Validate structure (grammar only)
        print("STEP 1: Validating structure and grammar...")
        validation = validate_structure_only(request)
        print("✓ Validation complete\n")
        
        # STEP 2: Find correct answer using UPDATED source priority
        print(f"{'='*80}")
        print("STEP 2: Finding Correct Answer")
        print(f"{'='*80}")
        
        final_answer = None
        options = [request.option1, request.option2, request.option3, request.option4]
        
        # SOURCE 1: From explanation (if provided - regardless of validation)
        if has_exp:
            print("\n→ SOURCE 1: Explanation (provided)")
            final_answer = get_answer_from_explanation(request.explanation, request.question, options)
            
            if final_answer:
                print("✓ SOURCE 1 SUCCESS")
            else:
                print("✗ SOURCE 1 FAILED (could not extract answer)")
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
        
        # SOURCE 3: From GPT-4 knowledge base
        if not final_answer:
            print("\n→ SOURCE 3: GPT-4 Knowledge Base (using OpenAI API)")
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
                    valid=validation.get('option1_valid', True),
                    feedback=validation.get('option1_feedback', '')
                ),
                option2=OptionValidation(
                    valid=validation.get('option2_valid', True),
                    feedback=validation.get('option2_feedback', '')
                ),
                option3=OptionValidation(
                    valid=validation.get('option3_valid', True),
                    feedback=validation.get('option3_feedback', '')
                ),
                option4=OptionValidation(
                    valid=validation.get('option4_valid', True),
                    feedback=validation.get('option4_feedback', '')
                ),
                option5=OptionValidation(
                    valid=validation.get('option5_valid', True),
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
    print("Source Priority:")
    print("  1. Explanation (if provided)")
    print("  2. Dataset (explanation first, then answer number)")
    print("  3. GPT-4 Knowledge Base (OpenAI API)")
    print("  4. Trusted News Sources (Prothom Alo, The Daily Star, BBC Bangla, etc.)")
    print(f"{'='*80}\n")
    
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)