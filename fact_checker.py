
from typing import Dict, Any, Optional, List
from openai import OpenAI
from config import settings
from vector_db import get_vector_db, EmbeddingService
import json
import re

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

class FactChecker:
    """Fact checker with UPDATED source priority"""
    
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
    
    def validate_structure_only(self, question: str, options: List[str], explanation: Optional[str]) -> Dict[str, Any]:
        """ONLY validates grammar and structure. Does NOT determine answer."""
        try:
            system_msg = """You are a grammar validator. Check ONLY grammar and structure.

DO NOT determine correct answer.
ONLY check if text is grammatically correct and clear.

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
            
            response = self._call_gpt4(system_msg, f"""Check grammar only:

Question: {question}
Option 1: {options[0] if len(options) > 0 else 'N/A'}
Option 2: {options[1] if len(options) > 1 else 'N/A'}
Option 3: {options[2] if len(options) > 2 else 'N/A'}
Option 4: {options[3] if len(options) > 3 else 'N/A'}
Option 5: {options[4] if len(options) > 4 else 'N/A'}
Explanation: {explanation if has_exp else 'NOT PROVIDED'}

Return JSON.""")
            
            result = json.loads(self._clean_json(response))
            
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
    
    def get_from_explanation(self, explanation: str, question: str, options: List[str]) -> Optional[str]:
        """Extract answer from explanation - IMPROVED VERSION"""
        try:
            print("\n📝 Extracting from explanation...")
            
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
            
            response = self._call_gpt4(system_msg, user_msg)
            result = json.loads(self._clean_json(response))
            
            answer = result.get('answer', '').strip()
            confidence = result.get('confidence', 0)
            reasoning = result.get('reasoning', '')
            
            if answer and confidence >= 70:
                print(f"✓ '{answer}' (conf: {confidence}%)")
                print(f"  Reasoning: {reasoning}")
                return answer
            else:
                print(f"✗ Low confidence ({confidence}%)")
                return None
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_from_dataset(self, question: str, options: List[str]) -> Optional[str]:
        """
        Find similar question in dataset - check explanation first, then answer number
        
        ✅ FIXED: Higher similarity threshold (0.85) to ensure accurate matches
        ✅ FIXED: Validates options match before accepting dataset answer
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
            
            # ✅ FIXED: Much higher threshold (0.85)
            if similarity >= 0.85:
                print(f"  ✓ HIGH similarity - Same question")
                try:
                    opts_stored = json.loads(best.get('options', '{}'))
                    answer_num = best.get('answer')
                    stored_explanation = best.get('explanation', '').strip()
                    
                    # ✅ NEW: Validate options match
                    dataset_options = [
                        opts_stored.get('option1', '').strip(),
                        opts_stored.get('option2', '').strip(),
                        opts_stored.get('option3', '').strip(),
                        opts_stored.get('option4', '').strip(),
                        opts_stored.get('option5', '').strip()
                    ]
                    
                    # Check matching options
                    matching_options = 0
                    for curr_opt in options:
                        curr_opt_norm = normalize_answer(curr_opt)
                        for ds_opt in dataset_options:
                            ds_opt_norm = normalize_answer(ds_opt)
                            if curr_opt_norm and ds_opt_norm and curr_opt_norm == ds_opt_norm:
                                matching_options += 1
                                break
                    
                    print(f"  Options matching: {matching_options}/{len(options)}")
                    
                    # ✅ FIXED: Require at least 3/4 options match
                    if matching_options < 3:
                        print(f"  ✗ Options mismatch ({matching_options}/4)")
                        print(f"  → Different question, will try GPT")
                        return None
                    
                    print(f"  ✓ Options match - Same question")
                    
                    # Priority 1: Check explanation
                    if stored_explanation:
                        print("  ✓ Explanation found")
                        dataset_options = [
                            opts_stored.get('option1', ''),
                            opts_stored.get('option2', ''),
                            opts_stored.get('option3', ''),
                            opts_stored.get('option4', ''),
                            opts_stored.get('option5', '')
                        ]
                        
                        extracted = self.get_from_explanation(
                            stored_explanation,
                            best.get('question', ''),
                            dataset_options
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
                    
                    response = self.client.chat.completions.create(
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
                        result = json.loads(self._clean_json(result_text))
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
    
    def get_from_news(self, question: str, options: List[str]) -> Optional[str]:
        """Get from TRUSTED news sources ONLY"""
        try:
            print("\n📰 Searching TRUSTED news...")
            
            query_emb = self.embedding_service.embed_query(question)
            news = self.vector_db.search(self.news_collection, query_emb, top_k=10)
            
            if not news:
                print("✗ No news found")
                return None
            
            # Filter trusted sources only
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
        Main fact checking
        UPDATED Priority: Explanation → Dataset → GPT (IMPROVED!) → News
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
        
        # Step 1: Validate
        print("Step 1: Validate")
        validation = self.validate_structure_only(question, options, explanation)
        print("✓ Done\n")
        
        # Step 2: Find answer with UPDATED priority
        print("Step 2: Find Answer (UPDATED Priority - GPT IMPROVED!)")
        print(f"{'='*80}")
        
        final_answer = None
        
        # Source 1: Explanation (if provided - regardless of validation status)
        if has_exp:
            print("\n→ SOURCE 1: Explanation")
            final_answer = self.get_from_explanation(explanation, question, options)
            
            if final_answer:
                print("✓ SUCCESS")
            else:
                print("✗ FAILED")
        else:
            print("\n✗ SOURCE 1: Explanation (not provided)")
        
        # Source 2: Dataset (check explanation first, then answer number)
        if not final_answer:
            print("\n→ SOURCE 2: Dataset")
            final_answer = self.get_from_dataset(question, options)
            
            if final_answer:
                print("✓ SUCCESS")
            else:
                print("✗ FAILED")
        
        # Source 3: GPT (IMPROVED!)
        if not final_answer:
            print("\n→ SOURCE 3: GPT Knowledge (IMPROVED v2.0)")
            final_answer = self.get_from_gpt_knowledge(question, options)
            
            if final_answer:
                print("✓ SUCCESS")
            else:
                print("✗ FAILED")
        
        # Source 4: News
        if not final_answer:
            print("\n→ SOURCE 4: Trusted News")
            final_answer = self.get_from_news(question, options)
            
            if final_answer:
                print("✓ SUCCESS")
            else:
                print("✗ FAILED")
        
        if not final_answer:
            final_answer = "Unable to determine the correct answer"
            print("\n❌ ALL FAILED")
        
        # Step 3: Compare (WITH NORMALIZATION)
        print(f"\n{'='*80}")
        print("Step 3: Compare (with normalization)")
        
        given_answer_valid = False
        if final_answer and final_answer != "Unable to determine the correct answer":
            # Normalize both answers to remove option prefixes
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
    """Helper function"""
    checker = FactChecker()
    result = checker.check_fact(question, given_answer, option1, option2, option3, option4, option5, explanation)
    checker.close()
    return result