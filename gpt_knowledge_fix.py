
"""
IMPROVED GPT KNOWLEDGE BASE FUNCTION
Replace the get_answer_from_gpt_knowledge function in backend.py with this version
"""

def get_answer_from_gpt_knowledge(question: str, options: List[str]) -> Optional[str]:
    """
    IMPROVED: Get answer from GPT-4 knowledge base with better accuracy
    Uses gpt-4o model and improved prompting strategy
    """
    try:
        print("\n🧠 Asking GPT-4 Knowledge Base (Improved)...")
        
        # IMPROVEMENT 1: More assertive and clearer system message
        system_msg = """You are a highly knowledgeable expert assistant. Your task is to answer the multiple choice question using your extensive training knowledge.

INSTRUCTIONS:
1. Read the question carefully
2. Analyze each option
3. Select the CORRECT answer based on your knowledge
4. Be confident - if you know the answer from your training, state it clearly
5. Return the answer as the EXACT TEXT from one of the provided options

You have knowledge about: history, science, geography, mathematics, current events, Bangladesh, world affairs, and many other topics. Use this knowledge confidently.

Return response in this JSON format:
{
    "answer": "exact text of the correct option",
    "confidence": 95,
    "reasoning": "brief explanation of why this is correct"
}

IMPORTANT: 
- Return the answer as EXACT TEXT from the options list
- Be confident in your knowledge
- If you truly don't know, set confidence to 0
- But if you have knowledge about the topic, answer confidently"""

        opts = "\n".join([f"{i+1}. {o}" for i, o in enumerate(options) if o])
        
        # IMPROVEMENT 2: Better user message with clearer instructions
        user_msg = f"""Question: {question}

Available Options:
{opts}

Based on your training knowledge, which option is correct? 

Analyze the question and provide your answer. Return ONLY valid JSON with your answer, confidence level, and reasoning."""

        # IMPROVEMENT 3: Try with gpt-4o first (better model), fallback to gpt-4-turbo
        models_to_try = ["gpt-4o", "gpt-4-turbo", "gpt-4"]
        
        for model in models_to_try:
            try:
                print(f"   Trying model: {model}")
                
                response = openai_client.chat.completions.create(
                    model=model,
                    temperature=0.3,  # Slightly higher for better reasoning
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ]
                )
                
                result_text = response.choices[0].message.content
                print(f"   Raw response: {result_text[:200]}...")
                
                # Parse JSON response
                result = json.loads(clean_json(result_text))
                answer = result.get('answer', '').strip()
                confidence = result.get('confidence', 0)
                reasoning = result.get('reasoning', '')
                
                # IMPROVEMENT 4: Better answer matching with normalization
                if answer:
                    print(f"   GPT says: '{answer}' (confidence: {confidence}%)")
                    print(f"   Reasoning: {reasoning}")
                    
                    # Try exact match first
                    for opt in options:
                        if opt.strip().lower() == answer.lower():
                            print(f"✓ GPT Knowledge ({model}): '{opt}' - EXACT MATCH")
                            print(f"  Confidence: {confidence}%")
                            print(f"  Reasoning: {reasoning}")
                            return opt
                    
                    # IMPROVEMENT 5: Try partial match if exact match fails
                    # Sometimes GPT might include extra text
                    answer_normalized = normalize_answer(answer)
                    for opt in options:
                        opt_normalized = normalize_answer(opt)
                        if answer_normalized in opt_normalized or opt_normalized in answer_normalized:
                            print(f"✓ GPT Knowledge ({model}): '{opt}' - PARTIAL MATCH")
                            print(f"  Confidence: {confidence}%")
                            print(f"  Reasoning: {reasoning}")
                            return opt
                    
                    # IMPROVEMENT 6: If confidence is high but no match, try to extract
                    if confidence >= 60:
                        # Check if answer contains any option text
                        for opt in options:
                            if opt.strip().lower() in answer.lower():
                                print(f"✓ GPT Knowledge ({model}): '{opt}' - EXTRACTED FROM ANSWER")
                                print(f"  Confidence: {confidence}%")
                                return opt
                        
                        # Still no match but high confidence - return GPT's answer anyway
                        print(f"⚠ GPT Knowledge ({model}): '{answer}' - HIGH CONFIDENCE BUT NO EXACT MATCH")
                        print(f"  Returning GPT's answer as-is")
                        return answer
                    
                    print(f"✗ GPT gave answer but low confidence: {confidence}%")
                
                # If this model worked (no exception), break the loop
                if confidence > 0:
                    break
                    
            except Exception as model_error:
                print(f"   ✗ Model {model} failed: {model_error}")
                continue
        
        # IMPROVEMENT 7: If all models failed or low confidence, try one more time with direct approach
        print("\n   Trying direct approach...")
        try:
            direct_system = """You are an expert. Answer this multiple choice question based on your knowledge. 
Just tell me which option is correct and why. Be direct and confident."""
            
            direct_user = f"""Question: {question}

Options:
{opts}

Which option is correct? Give me the option text and explain why briefly."""

            response = openai_client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": direct_system},
                    {"role": "user", "content": direct_user}
                ]
            )
            
            direct_answer = response.choices[0].message.content
            print(f"   Direct response: {direct_answer[:300]}")
            
            # Extract answer from natural language response
            for opt in options:
                if opt.strip().lower() in direct_answer.lower():
                    print(f"✓ GPT Knowledge (direct): Found '{opt}' in response")
                    return opt
            
        except Exception as direct_error:
            print(f"   ✗ Direct approach failed: {direct_error}")
        
        print("✗ GPT Knowledge: Could not determine answer with sufficient confidence")
        return None
    
    except Exception as e:
        print(f"✗ GPT Knowledge error: {e}")
        import traceback
        traceback.print_exc()
        return None


# Also update the call_gpt4 function to support multiple models
def call_gpt4(system_message: str, user_message: str, model: str = "gpt-4o") -> str:
    """
    IMPROVED: Helper function to call GPT-4 with better model selection
    """
    try:
        response = openai_client.chat.completions.create(
            model=model,  # Now supports model parameter
            temperature=0,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback to older model if needed
        if model == "gpt-4o":
            print(f"   Falling back to gpt-4-turbo...")
            response = openai_client.chat.completions.create(
                model="gpt-4-turbo",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.choices[0].message.content
        raise e