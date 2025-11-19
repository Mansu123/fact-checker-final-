
"""
Dataset Verification Script
Check if failing questions exist in your dataset
"""

import json
import sys

# Failing test questions
FAILING_QUESTIONS = [
    "দ্বৈত শাসনের প্রবর্তক কে?",
    "বাংলাদেশ সংবিধানের কোন ভাগে মৌলিক অধিকারের কথা বলা হয়েছে?",
    "রাষ্ট্রপতি সংবিধান সংশোধন বিল কত দিনের মধ্যে পাশ করবেন?"
]

def check_dataset(dataset_path):
    """Check if questions exist in dataset"""
    print("="*80)
    print("DATASET VERIFICATION")
    print("="*80)
    print(f"Dataset: {dataset_path}\n")
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ Loaded {len(data)} questions from dataset\n")
        
        for test_q in FAILING_QUESTIONS:
            print(f"\n{'='*80}")
            print(f"Searching for: {test_q}")
            print(f"{'='*80}")
            
            found = False
            for item in data:
                question = item.get('question', '')
                
                # Exact match
                if question == test_q:
                    found = True
                    print(f"✓ EXACT MATCH FOUND!")
                    print(f"  ID: {item.get('id')}")
                    print(f"  Answer option: {item.get('answer')}")
                    
                    answer_num = item.get('answer')
                    if answer_num:
                        answer_text = item.get(f'option{answer_num}', '')
                        print(f"  Answer text: {answer_text}")
                    
                    explanation = item.get('explain', '')
                    if explanation:
                        print(f"  Explanation: {explanation[:100]}...")
                    
                    break
                
                # Partial match (contains key words)
                elif any(word in question for word in test_q.split() if len(word) > 3):
                    if not found:  # Only show first partial match
                        print(f"⚠ PARTIAL MATCH:")
                        print(f"  Question: {question}")
                        print(f"  ID: {item.get('id')}")
                        found = "partial"
            
            if not found:
                print(f"✗ NOT FOUND in dataset")
                print(f"  Action needed: Add this question to your dataset")
            elif found == "partial":
                print(f"  This is not an exact match - might not be retrieved")
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print("If questions are NOT FOUND:")
        print("1. Add them to your dataset file")
        print("2. Make sure 'answer' field is correct")
        print("3. Re-run: python data_preprocessing.py")
        print("4. Restart API")
        print(f"{'='*80}\n")
        
    except FileNotFoundError:
        print(f"✗ Dataset file not found: {dataset_path}")
        print(f"  Check your path in .env file")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/favourite_question_40k_fixed.json"
    check_dataset(dataset_path)