
# Fact Checker & Question Generator - Enhanced v3.0

## 🎯 What's Fixed & New

### ✅ Fixed Issues

1. **Question Generator JSON Parsing Error**
   - Fixed the KeyError: '\n    "question"' error
   - Improved prompt formatting to avoid template variable conflicts
   - Better JSON extraction and validation
   - More robust error handling

2. **Topic Matching Improved**
   - Questions now properly match the requested topic (e.g., "হ্যালির ধূমকেতু")
   - Better similarity scoring and filtering
   - Paraphrasing now generates truly unique questions

### 🆕 New Features

1. **News Verification Integration**
   - Fact checker now verifies answers against recent news articles
   - Shows relevant news sources with publication dates
   - Helps identify if information has changed recently

2. **Math Problem Solver**
   - Automatically detects math questions
   - Uses GPT-4 to solve mathematical problems step-by-step
   - Supports both Bangla and English math questions
   - Returns detailed solution and explanation

3. **Enhanced API Schema**
   - Updated request/response models with individual option fields (option1-5)
   - Better validation and error messages
   - More detailed debug information

## 📋 Key Changes Summary

### Backend.py Changes

```python
# NEW: Individual option fields instead of options dict
class FactCheckRequest(BaseModel):
    question: str
    answer: str
    option1: Optional[str] = None  # NEW
    option2: Optional[str] = None  # NEW
    option3: Optional[str] = None  # NEW
    option4: Optional[str] = None  # NEW
    option5: Optional[str] = None  # NEW
    explanation: Optional[str] = None
    language: Optional[str] = "auto"

# NEW: News verification in response
class FactCheckResponse(BaseModel):
    # ... existing fields ...
    news_verification: Optional[Dict[str, Any]] = None  # NEW
    math_solution: Optional[Dict[str, Any]] = None      # NEW

# NEW: Individual option fields in response
class QuestionResponse(BaseModel):
    question: str
    option1: str      # NEW
    option2: str      # NEW
    option3: str      # NEW
    option4: str      # NEW
    option5: Optional[str] = ""  # NEW
    correct_answer: int
    explanation: str
```

### Question Generator Changes

```python
# FIXED: Avoid template variable conflicts
def paraphrase_question(self, original: Dict[str, Any]):
    # Build options without curly braces
    options_list = []
    for k, v in options.items():
        if v:
            options_list.append(f"{k}: {v}")
    options_text = "\n".join(options_list)
    
    # Use direct message creation instead of templates
    from langchain_core.messages import HumanMessage, SystemMessage
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ]
    
    response = self.llm.invoke(messages)
```

## 🚀 Installation

### Prerequisites

```bash
# Python 3.11 on macOS M1/M2
python3.11 -m venv venv311
source venv311/bin/activate
```

### Install Dependencies

```bash
# Core dependencies
pip install fastapi uvicorn pydantic pydantic-settings
pip install langchain langchain-openai langchain-core
pip install opensearch-py
pip install sentence-transformers torch
pip install requests beautifulsoup4 feedparser
pip install numpy tqdm

# For M1/M2 GPU support (optional but recommended)
pip install --upgrade torch torchvision
```

### Setup OpenSearch

```bash
# Start OpenSearch with Docker
docker-compose up -d opensearch

# Wait 1-2 minutes for OpenSearch to start
# Test connection
python test_opensearch.py
```

### Configure Environment

Create `.env` file:

```env
# OpenAI API Key
OPENAI_API_KEY=your-openai-api-key-here

# OpenSearch Configuration
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=admin12345678
OPENSEARCH_USE_SSL=true

# Vector DB Selection
VECTOR_DB=opensearch

# MySQL (optional)
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=dummy_password
MYSQL_DATABASE=fact_check_db

# Dataset
DATASET_PATH=/path/to/your/favourite_question_40k.json

# API
API_HOST=0.0.0.0
API_PORT=8000

# Embedding Configuration
EMBEDDING_TYPE=local
EMBEDDING_MODEL=text-embedding-3-small
LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
USE_GPU=true
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 📊 Usage

### 1. Preprocess Data

```bash
# Load dataset and create embeddings
python data_preprocessing.py
```

Expected output:
```
======================================================================
DATA PREPROCESSING FOR FACT CHECKER
======================================================================
Dataset: /path/to/favourite_question_40k.json
Vector DB: opensearch
Embedding: local
======================================================================
✓ Loaded 40000 questions
✓ Prepared 40000 documents
✓ Using ONLY questions for embeddings (not options/explanations)
✓ Generated 40000 embeddings
✓ Created OpenSearch index: fact_check_questions with dimension: 768
✓ Added all 40000 documents to OpenSearch
✓ Storage verification successful!
======================================================================
```

### 2. Start Backend Server

```bash
# Run the API server
python backend.py

# Or with auto-reload for development
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Test the API

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Fact Check (Simple)
```bash
curl -X POST "http://localhost:8000/fact-check" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "বাংলাদেশের রাজধানী কোথায়?",
    "answer": "1",
    "option1": "ঢাকা",
    "option2": "চট্টগ্রাম",
    "option3": "সিলেট",
    "option4": "রাজশাহী",
    "language": "bn"
  }'
```

#### Fact Check (Math Question)
```bash
curl -X POST "http://localhost:8000/fact-check" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is 15 + 27?",
    "answer": "42",
    "option1": "40",
    "option2": "42",
    "option3": "44",
    "option4": "45",
    "language": "en"
  }'
```

Response will include:
```json
{
  "is_correct": true,
  "confidence": 100.0,
  "stored_answer": "Option 2: 42",
  "user_answer": "42",
  "explanation": "...",
  "question_found": true,
  "similar_questions": [...],
  "language_detected": "en",
  "news_verification": null,
  "math_solution": {
    "solved": true,
    "solution": "15 + 27 = 42",
    "final_answer": "42",
    "explanation": "Simple addition",
    "method": "gpt4"
  }
}
```

#### Generate Questions
```bash
curl -X POST "http://localhost:8000/generate-questions" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "হ্যালির ধূমকেতু",
    "num_questions": 3,
    "language": "bn"
  }'
```

Response:
```json
[
  {
    "question": "হ্যালির ধূমকেতু সম্পর্কে পুনরায় প্রকাশিত প্রশ্ন...",
    "option1": "বিকল্প ১",
    "option2": "বিকল্প ২",
    "option3": "বিকল্প ৩",
    "option4": "বিকল্প ৪",
    "option5": "",
    "correct_answer": 1,
    "explanation": "ব্যাখ্যা..."
  }
]
```

### 4. Optional: Setup News Collection

```bash
# Run news collection agent (collects daily news)
python news_agent.py
```

Set up a cron job for daily collection:
```bash
# Edit crontab
crontab -e

# Add this line (runs daily at 2 AM)
0 2 * * * cd /path/to/project && /path/to/venv311/bin/python news_agent.py
```

## 🔍 API Endpoints

### POST /fact-check
Check if an answer is correct

**Request Body:**
```json
{
  "question": "string",
  "answer": "string",
  "option1": "string",
  "option2": "string",
  "option3": "string",
  "option4": "string",
  "option5": "string (optional)",
  "explanation": "string (optional)",
  "language": "auto|bn|en"
}
```

**Response:**
```json
{
  "is_correct": boolean,
  "confidence": float,
  "stored_answer": "string",
  "user_answer": "string",
  "explanation": "string",
  "question_found": boolean,
  "similar_questions": [...],
  "language_detected": "string",
  "news_verification": {
    "verified": boolean,
    "confidence": float,
    "message": "string",
    "sources": [...]
  },
  "math_solution": {
    "solved": boolean,
    "solution": "string",
    "final_answer": "string",
    "explanation": "string"
  }
}
```

### POST /generate-questions
Generate unique paraphrased questions on a topic

**Request Body:**
```json
{
  "topic": "string",
  "num_questions": 5,
  "language": "auto|bn|en"
}
```

**Response:**
```json
[
  {
    "question": "string",
    "option1": "string",
    "option2": "string",
    "option3": "string",
    "option4": "string",
    "option5": "string",
    "correct_answer": number,
    "explanation": "string"
  }
]
```

### POST /batch-verify
Verify multiple questions at once

### GET /health
Health check endpoint

### GET /debug/collection-info
Debug information about the database

## 🧪 Testing

### Test Question Generator
```bash
python question_generator.py
```

### Test Fact Checker
```bash
# Through API (server must be running)
curl http://localhost:8000/docs
```

## 🐛 Troubleshooting

### Issue: Question generator returns "KeyError"
**Solution:** This is now fixed in the new version. The issue was with template variable formatting.

### Issue: Questions don't match the topic
**Solution:** The similarity threshold has been adjusted. Questions now properly match the requested topic.

### Issue: News verification not working
**Solution:** 
1. Make sure you've run `news_agent.py` at least once
2. The news collection might be empty initially
3. Run the news agent and wait for it to collect articles

### Issue: Math questions not being solved
**Solution:**
1. Ensure your OpenAI API key is valid
2. Check that GPT-4 access is enabled on your account
3. Math detection requires numbers or operators in the question

### Issue: OpenSearch connection fails
**Solution:**
```bash
# Check if OpenSearch is running
docker ps | grep opensearch

# View logs
docker logs opensearch | tail -50

# Restart if needed
docker-compose restart opensearch

# Wait 1-2 minutes and test again
python test_opensearch.py
```

## 📈 Performance Tips

1. **Use Local Embeddings with GPU**
   - Set `EMBEDDING_TYPE=local` and `USE_GPU=true`
   - Much faster than OpenAI API (especially for M1/M2 Macs)
   - Free (no API costs)

2. **Batch Processing**
   - Use `/batch-verify` endpoint for multiple questions
   - More efficient than individual requests

3. **Caching**
   - Similar questions are cached by OpenSearch
   - Response times improve after initial queries

## 🔒 Security Notes

1. **API Key Security**
   - Never commit `.env` file to git
   - Use environment variables in production
   - Rotate API keys regularly

2. **OpenSearch Security**
   - Change default password in production
   - Enable SSL/TLS
   - Use firewall rules to restrict access

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📧 Support

For issues or questions, please check the troubleshooting section above.