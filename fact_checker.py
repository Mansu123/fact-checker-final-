
"""
Fact Checker Module
Complete separation: Validation vs Answer Finding
"""
from typing import Dict, Any, Optional, List
from openai import OpenAI
from config import settings
from vector_db import get_vector_db, EmbeddingService
import json
import re

class FactChecker:
    """Fact checker with clear source priority"""
    
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
                "options_consistency_valid": True, "options_consistency_feedback": "",
                "explanation_valid": False, "explanation_feedback": "Not provided"
            }
    
    def get_from_explanation(self, explanation: str, question: str, options: List[str]) -> Optional[str]:
        """Extract answer from explanation"""
        try:
            print("\n📝 Extracting from explanation...")
            
            system_msg = """Based on explanation, determine correct answer.
Return ONLY JSON:
{"answer": "correct answer as TEXT from options"}"""
            opts = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
            
            response = self._call_gpt4(system_msg, f"Question: {question}\nOptions:\n{opts}\n\nExplanation: {explanation}\n\nAnswer? Return ONLY JSON.")
            
            result = json.loads(self._clean_json(response))
            answer = result.get('answer', '').strip()
            
            if answer:
                print(f"✓ '{answer}'")
                return answer
            
            return None
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def get_from_gpt_knowledge(self, question: str, options: List[str]) -> Optional[str]:
        """Get from GPT-4 knowledge base using OpenAI API"""
        try:
            print("\n🧠 Asking GPT-4...")
            
            system_msg = """You are an expert. Answer based on your knowledge.
IMPORTANT: Only answer if confident. If unsure, return confidence 0.

Return ONLY JSON:
{
    "answer": "correct answer as TEXT from options",
    "confidence": 90,
    "reasoning": "brief explanation"
}
If unknown: {"answer": "", "confidence": 0, "reasoning": "Unknown"}"""
            opts = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
            
            response = self._call_gpt4(system_msg, f"Question: {question}\n\nOptions:\n{opts}\n\nAnswer based on your knowledge. Return ONLY JSON.")
            
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
            return None
    
    def get_from_news(self, question: str, options: List[str]) -> Optional[str]:
        """Get from TRUSTED news sources ONLY"""
        try:
            print("\n📰 Searching TRUSTED news...")
            
            query_emb = self.embedding_service.embed_query(question)
            news = self.vector_db.search(self.news_collection, query_emb, top_k=10)
            
            if not news:
                print(f"✗ No news found")
                return None
            
            # Filter trusted sources only
            trusted_sources = ["Prothom Alo", "The Daily Star", "BBC Bangla", "Bangladesh Pratidin", "NCTB"]
            trusted_news = [n for n in news if any(source.lower() in n.get('source', '').lower() for source in trusted_sources)]
            
            if not trusted_news:
                print(f"✗ No trusted sources")
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
                print(f"✗ No confident answer")
                return None
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def get_from_dataset(self, question: str, options: List[str]) -> Optional[str]:
        """Find similar question in dataset - check explanation first, then answer number"""
        try:
            print("\n💾 Searching dataset...")
            
            query_emb = self.embedding_service.embed_query(question)
            results = self.vector_db.search(self.collection_name, query_emb, top_k=10)
            
            if not results:
                print(f"✗ No results")
                return None
            
            print(f"✓ Found {len(results)} similar")
            
            best = max(results, key=lambda x: x.get('score', 0))
            similarity = best.get('score', 0)
            
            print(f"  Best: {similarity:.4f}")
            print(f"  Q: {best.get('question', '')[:80]}...")
            
            if similarity >= 0.12:
                try:
                    opts_stored = json.loads(best.get('options', '{}'))
                    answer_num = best.get('answer')
                    stored_explanation = best.get('explanation', '').strip()
                    
                    # Priority 1: Check explanation
                    if stored_explanation:
                        print(f"  ✓ Explanation found")
                        dataset_options = [
                            opts_stored.get('option1', ''),
                            opts_stored.get('option2', ''),
                            opts_stored.get('option3', ''),
                            opts_stored.get('option4', '')
                        ]
                        
                        extracted = self.get_from_explanation(stored_explanation, best.get('question', ''), dataset_options)
                        
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
                            print(f"  ✗ No answer text")
                    else:
                        print(f"  ✗ No answer number")
                        
                except Exception as e:
                    print(f"  ✗ Extract error: {e}")
            else:
                print(f"  ✗ Too low ({similarity:.4f} < 0.12)")
            
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
        explanation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main fact checking
        Priority: Explanation → GPT → News → Dataset
        """
        print(f"\n{'='*80}")
        print(f"FACT CHECK")
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
        
        # Step 2: Find answer
        print("Step 2: Find Answer")
        print(f"{'='*80}")
        
        final_answer = None
        
        # Source 1: Explanation
        if has_exp and validation.get('explanation_valid', False):
            print("\n→ SOURCE 1: Explanation")
            final_answer = self.get_from_explanation(explanation, question, options)
            if final_answer:
                print("✓ SUCCESS")
            else:
                print("✗ FAILED")
        else:
            print(f"\n✗ SOURCE 1: Explanation ({'invalid' if has_exp else 'not provided'})")
        
        # Source 2: GPT
        if not final_answer:
            print("\n→ SOURCE 2: GPT Knowledge")
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
        
        # Source 4: Dataset
        if not final_answer:
            print("\n→ SOURCE 4: Dataset")
            final_answer = self.get_from_dataset(question, options)
            if final_answer:
                print("✓ SUCCESS")
            else:
                print("✗ FAILED")
        
        if not final_answer:
            final_answer = "Unable to determine the correct answer"
            print("\n❌ ALL FAILED")
        
        # Step 3: Compare
        print(f"\n{'='*80}")
        print("Step 3: Compare")
        
        given_answer_valid = False
        
        if final_answer and final_answer != "Unable to determine the correct answer":
            given_clean = given_answer.strip().lower()
            final_clean = final_answer.strip().lower()
            
            print(f"Given: '{given_clean}'")
            print(f"Correct: '{final_clean}'")
            
            given_answer_valid = (given_clean == final_clean)
            print(f"Match: {given_answer_valid}")
        else:
            print(f"✗ Cannot validate")
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
    explanation: Optional[str] = None
) -> Dict[str, Any]:
    """Helper function"""
    checker = FactChecker()
    result = checker.check_fact(question, given_answer, option1, option2, option3, option4, explanation)
    checker.close()
    return result