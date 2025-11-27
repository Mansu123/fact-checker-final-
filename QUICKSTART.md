
# Quick Start Guide

## 🚀 Getting Started with the Combined System

### Files Created

1. **`backend.py`** - FastAPI backend with all improvements
2. **`fact_checker.py`** - Core fact-checking logic
3. **`SYSTEM_OVERVIEW.md`** - Comprehensive documentation
4. **`IMPROVEMENTS.md`** - What was improved and why

---

## 📦 Installation

### Prerequisites
```bash
# Required packages (if not already installed)
pip install fastapi uvicorn openai pydantic --break-system-packages
```

### Configuration
Make sure your `config.py` has:
```python
class Settings:
    openai_api_key = "your-api-key"
    api_host = "0.0.0.0"
    api_port = 8000
```

And your `vector_db.py` provides:
```python
def get_vector_db():
    # Returns your OpenSearch vector database

class EmbeddingService:
    def embed_query(self, text: str):
        # Returns embedding for text
```

---

## 🎯 Quick Test

### 1. Start the Server
```bash
python backend.py
```

You should see:
```
================================================================================
🚀 Fact Checker & MCQ Validator API - PERFECT COMBINED VERSION
================================================================================
✅ CORRECT: Answer validation with normalization
✅ CORRECT: Question/logic validation (reasonable strictness)
✅ CORRECT: Explanation validation (math & facts)
✅ IMPROVED: GPT Knowledge Base (v2.0)
================================================================================
```

### 2. Test the API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Simple Fact Check:**
```bash
curl -X POST http://localhost:8000/fact-check \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is 2+2?",
    "answer": "4",
    "option1": "3",
    "option2": "4",
    "option3": "5",
    "option4": "6"
  }'
```

**With Explanation:**
```bash
curl -X POST http://localhost:8000/fact-check \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of Bangladesh?",
    "answer": "ক) Dhaka",
    "option1": "ক) Dhaka",
    "option2": "খ) Chittagong",
    "option3": "গ) Sylhet",
    "option4": "ঘ) Rajshahi",
    "explanation": "Dhaka is the capital and largest city of Bangladesh."
  }'
```

---

## 📊 Understanding the Response

### Response Fields Explained

```json
{
  "question_valid": true,           // ✅ Question is understandable
  "feedback": "",                    // Error message if invalid
  
  "logical_valid": true,             // ✅ Options match question type
  
  "options": {
    "option1": {"feedback": ""},     // Individual option validation
    "option2": {"feedback": ""},
    "option3": {"feedback": ""},
    "option4": {"feedback": ""},
    "option5": {"feedback": ""},
    "options_consistency_valid": true, // ✅ No duplicates
    "feedback": ""                     // Duplicate details if any
  },
  
  "explanation_valid": true,         // ✅ Explanation is correct
  
  "given_answer_valid": true,        // ✅ Given answer matches correct answer
  
  "final_answer": "Dhaka"            // The correct answer found
}
```

### What Each Field Means

**`question_valid`**: 
- `true` = Question is understandable and answerable
- `false` = Question is gibberish/incomplete

**`logical_valid`**:
- `true` = Options are appropriate for the question
- `false` = Options don't match question type (e.g., text options for math question)

**`options_consistency_valid`**:
- `true` = No duplicate options
- `false` = Some options are duplicates

**`explanation_valid`**:
- `true` = Explanation is mathematically/factually correct
- `false` = Explanation contains errors OR not provided

**`given_answer_valid`**:
- `true` = Given answer matches the correct answer (after normalization)
- `false` = Given answer is wrong OR couldn't determine correct answer

**`final_answer`**:
- The correct answer as determined by the system
- "Unable to determine the correct answer" if all sources failed

---

## 🔍 Testing Different Scenarios

### Scenario 1: Correct Answer with Different Format
```json
{
  "question": "What is the capital?",
  "answer": "ক) Dhaka",              // Has prefix
  "option1": "Dhaka",                 // No prefix
  "option2": "Chittagong",
  "option3": "Sylhet",
  "option4": "Rajshahi"
}
```
**Expected:** `given_answer_valid: true` (normalized match)

---

### Scenario 2: Question with Minor Grammar Issue
```json
{
  "question": "Who first introduce gold coins in subcontinent?",
  "answer": "Kushanas",
  "option1": "Guptas",
  "option2": "Kushanas",
  "option3": "Mauryas",
  "option4": "Delhi Sultans"
}
```
**Expected:** `question_valid: true` (understandable despite grammar)

---

### Scenario 3: Wrong Explanation
```json
{
  "question": "What is √25 + 20?",
  "answer": "25",
  "option1": "25",
  "option2": "30",
  "option3": "35",
  "option4": "45",
  "explanation": "√25 = 10, then 10+20=30"  // Math errors
}
```
**Expected:** 
- `explanation_valid: false` (contains math errors)
- `given_answer_valid: true` (answer is still correct)
- `final_answer: "25"` (correct despite wrong explanation)

---

### Scenario 4: Duplicate Options
```json
{
  "question": "What is the capital?",
  "answer": "Dhaka",
  "option1": "Dhaka",
  "option2": "Chittagong",
  "option3": "Dhaka",                // Duplicate!
  "option4": "Rajshahi"
}
```
**Expected:**
- `options_consistency_valid: false`
- `feedback: "Option 1 and Option 3 are duplicates (both have 'dhaka')"`

---

## 🐛 Debugging

### Check Logs

The system provides detailed logging:

```
================================================================================
🔍 FACT CHECK REQUEST
================================================================================
Question: What is the capital of Bangladesh?
Given Answer: ক) Dhaka
Language: en
Explanation: PROVIDED
================================================================================

STEP 1: Validating question structure, grammar, and options...
✓ Validation complete

================================================================================
STEP 2: Finding Correct Answer
================================================================================

→ SOURCE 1: INPUT Explanation
📝 Extracting answer from explanation...
  ✓ Extracted answer: 'Dhaka' (95%)
✓ SOURCE 1 SUCCESS

================================================================================
STEP 3: Validating Explanation Correctness
================================================================================
🔍 Validating explanation correctness...
  Validation: ✅ VALID (95%)
  Reasoning: Factually correct - Dhaka is indeed the capital

================================================================================
STEP 4: Comparing Given Answer with Correct Answer
================================================================================
Given (original): 'ক) Dhaka'
Given (normalized): 'dhaka'
Correct (original): 'Dhaka'
Correct (normalized): 'dhaka'
✓ MATCH: Given answer is CORRECT

================================================================================
FINAL RESULT
================================================================================
Correct Answer: Dhaka
Given Answer: ক) Dhaka
Given Answer Valid: True
================================================================================
```

### Common Issues

**Issue 1: "Unable to determine the correct answer"**
- All 4 sources failed to find the answer
- Check if:
  - Question is in dataset (40,000+ questions)
  - GPT has knowledge of the topic
  - News sources cover the topic

**Issue 2: "given_answer_valid: false" when answer looks correct**
- Check normalization
- Compare normalized values in logs
- Ensure option format is consistent

**Issue 3: "explanation_valid: false" but explanation looks correct**
- Check math calculations carefully
- Look at validation reasoning in logs
- Confidence threshold is 70%

---

## 🎯 Integration Examples

### Python Integration
```python
from fact_checker import check_fact

result = check_fact(
    question="What is the capital of Bangladesh?",
    given_answer="Dhaka",
    option1="Dhaka",
    option2="Chittagong",
    option3="Sylhet",
    option4="Rajshahi",
    explanation="Dhaka is the capital and largest city of Bangladesh."
)

if result['given_answer_valid']:
    print("✅ Correct answer!")
else:
    print(f"❌ Wrong answer. Correct: {result['final_answer']}")
```

### API Integration (JavaScript)
```javascript
async function checkFact(question, answer, options, explanation) {
  const response = await fetch('http://localhost:8000/fact-check', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question,
      answer,
      option1: options[0],
      option2: options[1],
      option3: options[2],
      option4: options[3],
      explanation
    })
  });
  
  const result = await response.json();
  
  if (result.given_answer_valid) {
    console.log("✅ Correct answer!");
  } else {
    console.log(`❌ Wrong. Correct: ${result.final_answer}`);
  }
  
  return result;
}
```

---

## 📈 Performance Tips

### 1. Caching
Consider caching results for frequently asked questions:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_fact_check(question_hash):
    # Your fact check logic
    pass
```

### 2. Batch Processing
Process multiple questions in parallel:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def check_multiple(questions):
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = await asyncio.gather(*[
            executor.submit(check_fact, q) 
            for q in questions
        ])
    return results
```

### 3. Database Optimization
Ensure your OpenSearch has:
- Proper indexing
- Optimized embeddings
- Regular maintenance

---

## 🎉 You're Ready!

Your combined system is now:
- ✅ Properly configured
- ✅ Tested and verified
- ✅ Ready for production use

For detailed documentation, see:
- `SYSTEM_OVERVIEW.md` - Complete system documentation
- `IMPROVEMENTS.md` - What was improved and why

Need help? Check the logs - they're very detailed! 🚀