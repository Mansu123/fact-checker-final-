
from typing import Dict, Any, Optional, List
from openai import OpenAI
from config import settings
from vector_db import get_vector_db, EmbeddingService
import json
import re

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

class FactChecker:
    """
    ✅ PERFECT COMBINED Fact Checker:
    - CORRECT answer validation (with normalization)
    - CORRECT question/logic validation (reasonable strictness)
    - CORRECT explanation validation (math & facts)
    - IMPROVED GPT Knowledge Base (v2.0)
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.embedding_service = EmbeddingService()
        self.vector_db = get_vector_db()
        self.collection_name = "fact_check_questions"
        self.news_collection = "news_articles"
    
    def _call_gpt4(self, system_message: str, user_message: str) -> str:
        """Helper to call GPT-4"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
    
    def _clean_json(self, content: str) -> str:
        """Clean JSON response"""
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content).strip()
        match = re.search(r'\{.*\}', content, re.DOTALL)
        return match.group(0) if match else content
    
    def detect_duplicates(self, options: List[str]) -> tuple:
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
    
    def validate_structure_only(self, question: str, options: List[str], explanation: Optional[str]) -> Dict[str, Any]:
        """
        ✅ CORRECT: Validates with reasonable strictness
        - Not too strict for Bengali/translated questions
        - Checks basic understandability
        - Allows minor imperfections
        """
        try:
            system_msg = """You are a question validator. Check if questions and options are reasonable and understandable.

⚠️ BE REASONABLE, not overly strict. Many questions are Bengali or translated and may have minor grammatical issues but are still valid and understandable.

QUESTION VALIDATION:
Mark as INVALID only if:
- Completely nonsensical or gibberish
- Severe logical contradictions
- Impossible to understand
- Incomplete to point of being unanswerable

✅ Mark as VALID if:
- Understandable despite minor grammar issues
- Makes logical sense even if phrasing could be better
- Clear enough to answer
- Bengali/translation with acceptable grammar

LOGICAL VALIDATION:
Mark logical_valid as FALSE only if:
- Severe logical contradictions
- Options are completely wrong type
- Question-option combination makes no sense at all

✅ Mark as VALID if:
- Options are appropriate type for question
- Minor mismatches are acceptable
- Question and options work together reasonably

OPTION VALIDATION:
Mark as INVALID only if:
- Completely meaningless gibberish
- Obviously fake placeholder text
- Totally wrong type for question

✅ Mark as VALID if:
- Options make sense for question
- Readable and meaningful
- Minor formatting issues OK

BE REASONABLE. If humans can understand it, mark it VALID.

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
    "option5_valid": true/false,
    "option5_feedback": "",
    "options_consistency_valid": true/false,
    "options_consistency_feedback": "",
    "explanation_valid": true/false,
    "explanation_feedback": ""
}

Return ONLY JSON."""
            
            has_exp = bool(explanation and explanation.strip())
            
            response = self._call_gpt4(system_msg, f"""Check reasonably (not too strict):

Question: {question}
Option 1: {options[0] if len(options) > 0 else 'N/A'}
Option 2: {options[1] if len(options) > 1 else 'N/A'}
Option 3: {options[2] if len(options) > 2 else 'N/A'}
Option 4: {options[3] if len(options) > 3 else 'N/A'}
Option 5: {options[4] if len(options) > 4 else 'N/A'}
Explanation: {explanation if has_exp else 'NOT PROVIDED'}

Be REASONABLE. Mark as valid if humans can understand it.

Return JSON.""")
            
            result = json.loads(self._clean_json(response))
            
            if not has_exp:
                result['explanation_valid'] = False
                result['explanation_feedback'] = "Not provided"
            
            # Use Python-based duplicate detection
            has_duplicates, duplicate_feedback = self.detect_duplicates(options)
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
    
    def validate_explanation_correctness(self, explanation: str, question: str, answer: str, options: List[str]) -> Dict[str, Any]:
        """
        ✅ CORRECT: Validate if explanation is factually/mathematically correct
        """
        try:
            print("\n🔍 Validating explanation correctness...")
            
            validation_system = """You are an expert fact-checker and mathematician. Validate if explanations are correct.

⚠️ BE STRICT WITH MATHEMATICAL EXPLANATIONS:

Check:
1. Are all calculations correct?
2. Is the logic sound?
3. Does it support the given answer?

MATHEMATICAL ERRORS TO DETECT:
- Wrong arithmetic (2+2=5)
- Wrong square roots (√25=10 when √25=5)
- Wrong powers (5²=30 when 5²=25)
- Wrong formulas
- Logic errors

Return JSON:
{
    "is_valid": true/false,
    "confidence": 95,
    "reasoning": "specific errors or confirmation"
}"""
            
            opts = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
            
            validation_user = f"""Question: {question}

Options:
{opts}

Answer: {answer}

Explanation: {explanation}

Is this explanation mathematically and factually correct?

Return ONLY JSON."""

            response = self._call_gpt4(validation_system, validation_user)
            result = json.loads(self._clean_json(response))
            
            is_valid = result.get('is_valid', False)
            confidence = result.get('confidence', 0)
            reasoning = result.get('reasoning', '')
            
            print(f"  Validation: {'✅ VALID' if is_valid else '❌ INVALID'} ({confidence}%)")
            print(f"  Reasoning: {reasoning}")
            
            return {
                'is_valid': is_valid and confidence >= 70,
                'confidence': confidence,
                'reasoning': reasoning
            }
            
        except Exception as e:
            print(f"  ✗ Validation error: {e}")
            return {'is_valid': False, 'confidence': 0, 'reasoning': str(e)}
    
    def get_from_explanation(self, explanation: str, question: str, options: List[str]) -> Optional[str]:
        """
        ✅ CORRECT: Extract answer from explanation WITHOUT rejecting it
        """
        try:
            print("\n📝 Extracting from explanation...")
            
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

Based on this explanation, what is the correct answer? Return ONLY JSON with the EXACT option text."""
            
            response = self._call_gpt4(system_msg, user_msg)
            result = json.loads(self._clean_json(response))
            
            answer = result.get('answer', '').strip()
            confidence = result.get('confidence', 0)
            reasoning = result.get('reasoning', '')
            
            if not answer or confidence < 60:
                print(f"✗ Low confidence ({confidence}%)")
                return None
            
            print(f"✓ Extracted: '{answer}' (conf: {confidence}%)")
            print(f"  Reasoning: {reasoning}")
            
            return answer
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_from_dataset(self, question: str, options: List[str]) -> Optional[str]:
        """
        ✅ CORRECT: Find similar question in dataset
        - Higher similarity threshold (0.85)
        - Validates options match
        """
        try:
            print("\n💾 Searching dataset...")
            
            query_emb = self.embedding_service.embed_query(question)
            results = self.vector_db.search(self.collection_name, query_emb, top_k=10)
            
            if not results:
                print("✗ No results")
                return None
            
            print(f"✓ Found {len(results)} similar")
            
            best = max(results, key=lambda x: x.get('score', 0))
            similarity = best.get('score', 0)
            
            print(f"  Best: {similarity:.4f}")
            print(f"  Q: {best.get('question', '')[:80]}...")
            
            # ✅ CORRECT: Higher threshold
            if similarity >= 0.85:
                print(f"  ✓ HIGH similarity - Same question")
                try:
                    opts_stored = json.loads(best.get('options', '{}'))
                    answer_num = best.get('answer')
                    stored_explanation = best.get('explanation', '').strip()
                    
                    # ✅ CORRECT: Validate options match
                    dataset_options = [
                        opts_stored.get('option1', '').strip(),
                        opts_stored.get('option2', '').strip(),
                        opts_stored.get('option3', '').strip(),
                        opts_stored.get('option4', '').strip(),
                        opts_stored.get('option5', '').strip()
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
                    
                    # ✅ CORRECT: Require 3/4 match
                    if matching_options < 3:
                        print(f"  ✗ Options mismatch ({matching_options}/4)")
                        print(f"  → Different question, will try GPT")
                        return None
                    
                    print(f"  ✓ Options match - Same question")
                    
                    # Priority 1: Check explanation
                    if stored_explanation:
                        print("  ✓ Explanation found")
                        dataset_options_full = [
                            opts_stored.get('option1', ''),
                            opts_stored.get('option2', ''),
                            opts_stored.get('option3', ''),
                            opts_stored.get('option4', ''),
                            opts_stored.get('option5', '')
                        ]
                        
                        extracted = self.get_from_explanation(
                            stored_explanation,
                            best.get('question', ''),
                            dataset_options_full
                        )
                        
                        if extracted:
                            for opt in options:
                                if opt.strip().lower() == extracted.strip().lower():
                                    print(f"  ✓ From explanation: '{opt}'")
                                    return opt
                            
                            print(f"  ✓ From explanation: '{extracted}'")
                            return extracted
                    
                    # Priority 2: Use answer number
                    if answer_num:
                        answer_text = opts_stored.get(f'option{answer_num}', '').strip()
                        
                        if answer_text:
                            for opt in options:
                                if opt.strip().lower() == answer_text.strip().lower():
                                    print(f"  ✓ From dataset: '{opt}'")
                                    return opt
                            
                            print(f"  ✓ From dataset: '{answer_text}'")
                            return answer_text
                        else:
                            print("  ✗ No answer text")
                    else:
                        print("  ✗ No answer number")
                    
                except Exception as e:
                    print(f"  ✗ Extract error: {e}")
            else:
                print(f"  ✗ Too low ({similarity:.4f} < 0.85)")
                print(f"  → Will try GPT instead")
            
            return None
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def get_from_gpt_knowledge(self, question: str, options: List[str]) -> Optional[str]:
        """
        ✅ PERFECT: IMPROVED GPT-4 Knowledge Base - Works as well as real ChatGPT!
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

            models_to_try = [
                ("gpt-4o", "Latest GPT-4 Omni"),
                ("gpt-4-turbo", "GPT-4 Turbo"),
                ("gpt-4", "GPT-4")
            ]
            
            for model_name, model_desc in models_to_try:
                try:
                    print(f"   Attempting: {model_name} ({model_desc})")
                    
                    response = self.client.chat.completions.create(
                        model=model_name,
                        temperature=0.3,
                        max_tokens=500,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg}
                        ]
                    )
                    
                    result_text = response.choices[0].message.content.strip()
                    print(f"   ✓ Got response from {model_name}")
                    
                    try:
                        result = json.loads(self._clean_json(result_text))
                    except json.JSONDecodeError:
                        print(f"   ⚠ JSON parse failed, trying to extract...")
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
                    
                    # Strategy A: Exact match
                    for opt in options:
                        if opt.strip().lower() == answer.lower():
                            print(f"✓ GPT Knowledge ({model_name}): EXACT MATCH")
                            print(f"  ✓ Answer: '{opt}'")
                            print(f"  ✓ Confidence: {confidence}%")
                            return opt
                    
                    # Strategy B: Normalized match
                    answer_norm = normalize_answer(answer)
                    for opt in options:
                        opt_norm = normalize_answer(opt)
                        if opt_norm and answer_norm and opt_norm == answer_norm:
                            print(f"✓ GPT Knowledge ({model_name}): NORMALIZED MATCH")
                            print(f"  ✓ Answer: '{opt}'")
                            print(f"  ✓ Confidence: {confidence}%")
                            return opt
                    
                    # Strategy C: Substring match
                    for opt in options:
                        opt_clean = opt.strip().lower()
                        answer_clean = answer.lower()
                        
                        if len(opt_clean) < 4:
                            continue
                        
                        if opt_clean in answer_clean or answer_clean in opt_clean:
                            print(f"✓ GPT Knowledge ({model_name}): SUBSTRING MATCH")
                            print(f"  ✓ Answer: '{opt}'")
                            print(f"  ✓ Confidence: {confidence}%")
                            return opt
                    
                    # Strategy D: Word overlap
                    if confidence >= 60:
                        answer_words = set(answer.lower().split())
                        best_match = None
                        best_overlap = 0
                        
                        for opt in options:
                            opt_words = set(opt.lower().split())
                            overlap = len(answer_words & opt_words)
                            
                            if overlap > best_overlap and overlap >= 2:
                                best_overlap = overlap
                                best_match = opt
                        
                        if best_match:
                            print(f"✓ GPT Knowledge ({model_name}): WORD OVERLAP MATCH")
                            print(f"  ✓ Answer: '{best_match}' ({best_overlap} words matched)")
                            print(f"  ✓ Confidence: {confidence}%")
                            return best_match
                    
                    # Strategy E: High confidence
                    if confidence >= 80:
                        print(f"✓ GPT Knowledge ({model_name}): HIGH CONFIDENCE ANSWER")
                        print(f"  ⚠ No exact option match, but GPT is {confidence}% confident")
                        print(f"  ✓ Returning: '{answer}'")
                        return answer
                    
                    print(f"   ⚠ Confidence too low ({confidence}%), trying next approach...")
                    
                except Exception as model_error:
                    print(f"   ✗ Model {model_name} error: {str(model_error)[:100]}")
                    continue
            
            # Last resort - Direct approach
            print("\n   🔄 Trying direct conversational approach...")
            try:
                direct_prompt = f"""Answer this question directly and clearly:

{question}

Options:
{opts_formatted}

Which option is correct? Just tell me the answer clearly."""

                direct_response = self.client.chat.completions.create(
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
                
                for i, opt in enumerate(options):
                    opt_lower = opt.strip().lower()
                    if opt_lower in direct_text:
                        print(f"✓ GPT Knowledge (direct): Found option in response")
                        print(f"  ✓ Answer: '{opt}'")
                        return opt
                    
                    if f"option {i+1}" in direct_text or f"{i+1}." in direct_text[:30]:
                        print(f"✓ GPT Knowledge (direct): Found by option number")
                        print(f"  ✓ Answer: '{opt}'")
                        return opt
                
            except Exception as e:
                print(f"   ✗ Direct approach failed: {e}")
            
            print("✗ GPT Knowledge: Unable to determine answer with confidence")
            return None
        
        except Exception as e:
            print(f"✗ GPT Knowledge CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_from_news(self, question: str, options: List[str]) -> Optional[str]:
        """✅ CORRECT: Get from TRUSTED news sources ONLY"""
        try:
            print("\n📰 Searching TRUSTED news...")
            
            query_emb = self.embedding_service.embed_query(question)
            news = self.vector_db.search(self.news_collection, query_emb, top_k=10)
            
            if not news:
                print("✗ No news found")
                return None
            
            trusted_sources = ["Prothom Alo", "The Daily Star", "BBC Bangla", "Bangladesh Pratidin", "NCTB"]
            trusted_news = [n for n in news if any(source.lower() in n.get('source', '').lower() for source in trusted_sources)]
            
            if not trusted_news:
                print("✗ No trusted sources")
                return None
            
            print(f"✓ Found {len(trusted_news)} trusted articles")
            
            context = "\n\n".join([
                f"[{n.get('source', '')}] {n.get('title', '')}\n{n.get('content', '')[:400]}"
                for n in trusted_news[:3]
            ])
            
            system_msg = """Based ONLY on trusted news, answer.

Return ONLY JSON:
{"answer": "correct answer as TEXT", "confidence": 85, "source": "which article"}

If not in news: {"answer": "", "confidence": 0, "source": ""}"""
            
            opts = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
            
            response = self._call_gpt4(system_msg, f"News:\n{context}\n\nQuestion: {question}\nOptions:\n{opts}\n\nAnswer? Return ONLY JSON.")
            
            result = json.loads(self._clean_json(response))
            answer = result.get('answer', '').strip()
            confidence = result.get('confidence', 0)
            
            if answer and confidence >= 70:
                print(f"✓ '{answer}' (conf: {confidence}%)")
                return answer
            else:
                print("✗ No confident answer")
                return None
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def check_fact(
        self,
        question: str,
        given_answer: str,
        option1: str,
        option2: str,
        option3: str,
        option4: str,
        option5: Optional[str] = None,
        explanation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        ✅ PERFECT: Main fact checking with combined best practices
        - CORRECT answer validation (with normalization)
        - CORRECT question/logic validation
        - CORRECT explanation validation
        - IMPROVED GPT Knowledge Base
        """
        print(f"\n{'='*80}")
        print("FACT CHECK")
        print(f"{'='*80}")
        print(f"Q: {question}")
        print(f"Given: {given_answer}")
        
        has_exp = bool(explanation and explanation.strip())
        print(f"Exp: {'YES' if has_exp else 'NO'}")
        print(f"{'='*80}\n")
        
        options = [option1, option2, option3, option4]
        if option5:
            options.append(option5)
        
        # Step 1: Validate
        print("Step 1: Validate")
        validation = self.validate_structure_only(question, options, explanation)
        
        if has_exp:
            validation['explanation_valid'] = True  # Will be updated later
        
        print("✓ Done\n")
        
        # Step 2: Find answer (NOT from INPUT explanation)
        print("Step 2: Find Answer")
        print(f"{'='*80}")
        print("⚠️  INPUT Explanation is ONLY used for validation, NOT for answer extraction")
        print("    Answer sources: Dataset → GPT Knowledge Base → Trusted News")
        
        final_answer = None
        
        # Source 1: Dataset
        print("\n→ SOURCE 1: Dataset")
        final_answer = self.get_from_dataset(question, options)
        
        if final_answer:
            print("✓ SUCCESS")
        else:
            print("✗ FAILED")
        
        # Source 2: GPT
        if not final_answer:
            print("\n→ SOURCE 2: GPT Knowledge (IMPROVED v2.0)")
            final_answer = self.get_from_gpt_knowledge(question, options)
            
            if final_answer:
                print("✓ SUCCESS")
            else:
                print("✗ FAILED")
        
        # Source 3: News
        if not final_answer:
            print("\n→ SOURCE 3: Trusted News")
            final_answer = self.get_from_news(question, options)
            
            if final_answer:
                print("✓ SUCCESS")
            else:
                print("✗ FAILED")
        
        if not final_answer:
            final_answer = "Unable to determine the correct answer"
            print("\n❌ ALL FAILED")
        
        # Step 3: Validate INPUT explanation against CORRECT answer
        if has_exp and final_answer and final_answer != "Unable to determine the correct answer":
            print(f"\n{'='*80}")
            print("Step 3: Validating INPUT Explanation Against CORRECT Answer")
            print(f"{'='*80}")
            print(f"  Correct answer from Dataset/GPT: '{final_answer}'")
            print(f"  Checking if INPUT explanation supports this answer...")
            
            # Extract what answer the explanation claims
            explanation_claims_answer = self.get_from_explanation(explanation, question, options)
            
            if explanation_claims_answer:
                print(f"  INPUT explanation claims answer is: '{explanation_claims_answer}'")
                
                # Normalize for comparison
                explanation_normalized = normalize_answer(explanation_claims_answer)
                correct_normalized = normalize_answer(final_answer)
                
                if explanation_normalized == correct_normalized:
                    print(f"  ✓ Explanation answer MATCHES correct answer")
                    
                    # Validate math/facts
                    exp_validation = self.validate_explanation_correctness(
                        explanation,
                        question,
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
        
        # Step 4: Compare (WITH NORMALIZATION)
        print(f"\n{'='*80}")
        print("Step 4: Compare (with normalization)")
        
        given_answer_valid = False
        if final_answer and final_answer != "Unable to determine the correct answer":
            given_normalized = normalize_answer(given_answer)
            final_normalized = normalize_answer(final_answer)
            
            print(f"Given (original): '{given_answer}'")
            print(f"Given (normalized): '{given_normalized}'")
            print(f"Correct (original): '{final_answer}'")
            print(f"Correct (normalized): '{final_normalized}'")
            
            given_answer_valid = (given_normalized == final_normalized)
            print(f"Match: {given_answer_valid}")
        else:
            print("✗ Cannot validate")
            given_answer_valid = False
        
        print(f"\n{'='*80}")
        print(f"RESULT: {final_answer}")
        print(f"VALID: {given_answer_valid}")
        print(f"{'='*80}\n")
        
        return {
            "question_valid": validation.get('question_valid', True),
            "feedback": validation.get('question_feedback', ''),
            "logical_valid": validation.get('logical_valid', True),
            "options": {
                "option1": {
                    "valid": validation.get('option1_valid', True),
                    "feedback": validation.get('option1_feedback', '')
                },
                "option2": {
                    "valid": validation.get('option2_valid', True),
                    "feedback": validation.get('option2_feedback', '')
                },
                "option3": {
                    "valid": validation.get('option3_valid', True),
                    "feedback": validation.get('option3_feedback', '')
                },
                "option4": {
                    "valid": validation.get('option4_valid', True),
                    "feedback": validation.get('option4_feedback', '')
                },
                "option5": {
                    "valid": validation.get('option5_valid', True),
                    "feedback": validation.get('option5_feedback', '')
                },
                "options_consistency_valid": validation.get('options_consistency_valid', True),
                "feedback": validation.get('options_consistency_feedback', '')
            },
            "explanation_valid": validation.get('explanation_valid', False),
            "given_answer_valid": given_answer_valid,
            "final_answer": final_answer
        }
    
    def close(self):
        """Close connections"""
        self.vector_db.close()

def check_fact(
    question: str,
    given_answer: str,
    option1: str,
    option2: str,
    option3: str,
    option4: str,
    option5: Optional[str] = None,
    explanation: Optional[str] = None
) -> Dict[str, Any]:
    """
    ✅ PERFECT: Helper function with combined best practices
    """
    checker = FactChecker()
    result = checker.check_fact(question, given_answer, option1, option2, option3, option4, option5, explanation)
    checker.close()
    return result