
MCQ Question Validation System
A comprehensive fact-checking and validation system for Multiple Choice Questions (MCQ), specifically designed for Bengali and English educational content, including Bangladesh Civil Service (BCS) exam questions.
Overview
This system validates MCQ questions through a multi-step process that checks grammar, relevance, and answer accuracy using multiple AI models, vector databases, and trusted data sources.
Features

Multi-language Support: Handles both Bengali and English questions
4-Step Validation Pipeline: Comprehensive validation from grammar to answer verification
Multiple Data Sources: Dataset search, AI knowledge bases, and web search
High-Performance: Average response time of 5-8 seconds
Cost-Optimized: Smart model routing reduces costs from $42 to $1-2 per 1,000 questions
Accurate Answer Detection: 85%+ similarity threshold for reliable matching

System Workflow
USER INPUT
    ↓
[Question + 4/5 Options + Given Answer + Optional Explanation]
    ↓
┌─────────────────────────────────────────────────────┐
│  STEP 1: VALIDATION (Grammar & Relevance)           │
│  ✓ Check question grammar                           │
│  ✓ Check each option grammar                        │
│  ✓ Check if options are RELEVANT to question        │
│  ✓ Check options consistency                        │
│  ✓ Check explanation (if provided)                  │
│    - Question logic test and explanation correct    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  STEP 2: FIND CORRECT ANSWER (4 Sources)            │
│                                                      │
│  📝 SOURCE 1: Dataset                               │
│     IF news not found                               │
│     → Find similar question in OpenSearch           │
│     → Check similarity score for question and       │
│       option ≥0.85                                  │
│     → Priority:                                     │
│       1. Extract from dataset's explanation         │
│       2. Use dataset's answer number (1,2,3,4)      │
│     → Convert to text option                        │
│     → If found → DONE                               │
│                                                      │
│  🧠 SOURCE 2: Gemini Flash Lite Knowledge Base      │
│     IF no explanation                               │
│     → Ask gemini-2.5-flash-lite for general         │
│       questions                                     │
│     → And GPT-4o mini for math questions            │
│       (better performance)                          │
│     → Confidence must be ≥70%                       │
│     → If confident → DONE                           │
│                                                      │
│  📰 SOURCE 3: Trusted News Sources (GPT Web Search) │
│     IF GPT-4 doesn't know                           │
│     → Search vector DB for news                     │
│     → Only use: Prothom Alo, The Daily Star,       │
│       BBC Bangla, Bangladesh Pratidin, NCTB,       │
│       Government websites                           │
│     → Confidence must be ≥70%                       │
│     → If found → DONE                               │
│                                                      │
│  ❌ FALLBACK: If all fail                           │
│     → "Unable to determine the correct answer"      │
│     → given_answer_valid = FALSE                    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  STEP 3: COMPARE ANSWERS                            │
│  1. Clean both answers (remove ক), খ), a), b))     │
│  2. Normalize: lowercase, trim spaces               │
│  3. Compare: given == correct                       │
│  4. Set: given_answer_valid = TRUE/FALSE            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  STEP 4: RETURN RESPONSE                            │
│  {                                                   │
│    question_valid: boolean                          │
│    logical_valid: boolean                           │
│    options: {                                       │
│      option1: {valid, feedback},                    │
│      option2: {valid, feedback},                    │
│      option3: {valid, feedback},                    │
│      option4: {valid, feedback},                    │
│      options_consistency_valid: boolean,            │
│      feedback: string                               │
│    },                                               │
│    explanation_valid: boolean                       │
│    given_answer_valid: boolean ← KEY RESULT         │
│    final_answer: "খোঁজখবর" ← CLEANED (no ক), খ))   │
│  }                                                   │
└─────────────────────────────────────────────────────┘
System Architecture
Input Format
json{
  "question": "Question text",
  "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
  "given_answer": "Option text",
  "explanation": "Optional explanation text"
}
Validation Pipeline
STEP 1: Validation (Grammar & Relevance)
Validates the structural integrity and logical consistency of the question:

✓ Question Grammar: Checks for grammatical correctness
✓ Options Grammar: Validates each option individually
✓ Relevance Check: Ensures options are relevant to the question
✓ Consistency Check: Verifies options follow consistent formatting
✓ Explanation Validation: Tests question logic and explanation accuracy (if provided)

STEP 2: Find Correct Answer (Multi-Source Approach)
The system uses a waterfall approach with 4 prioritized sources:
📝 SOURCE 1: Dataset Search (OpenSearch)

Searches 40,000+ pre-validated questions in vector database
Similarity Threshold: ≥0.85 for both question and options
Priority Order:

Extract answer from dataset's explanation
Use dataset's answer number (1, 2, 3, 4)
Convert to text option


Action: If found with high confidence → DONE

🧠 SOURCE 2: AI Knowledge Base

Model Routing:

Gemini 2.5 Flash Lite: General questions
GPT-4o Mini: Mathematical questions (better performance)


Confidence Threshold: ≥70%
Action: If confident answer found → DONE

📰 SOURCE 3: Trusted News Sources (GPT-4 Web Search)

Searches vector database for news-related questions
Trusted Sources Only:

Prothom Alo
The Daily Star
BBC Bangla
Bangladesh Pratidin
NCTB
Government websites


Confidence Threshold: ≥70%
Action: If found in trusted sources → DONE

❌ FALLBACK: Unable to Determine

Returns: "Unable to determine the correct answer"
Sets: given_answer_valid = FALSE

STEP 3: Answer Comparison
Compares the given answer with the system-determined correct answer:

Clean Answers: Remove option markers (ক), খ), a), b))
Normalize: Convert to lowercase, trim whitespace
Compare: given_answer == correct_answer
Result: Set given_answer_valid = TRUE/FALSE

STEP 4: Response Generation
Returns a comprehensive validation report:
json{
  "question_valid": boolean,
  "logical_valid": boolean,
  "options": {
    "option1": {
      "valid": boolean,
      "feedback": "string"
    },
    "option2": { "valid": boolean, "feedback": "string" },
    "option3": { "valid": boolean, "feedback": "string" },
    "option4": { "valid": boolean, "feedback": "string" },
    "options_consistency_valid": boolean,
    "feedback": "string"
  },
  "explanation_valid": boolean,
  "given_answer_valid": boolean,
  "final_answer": "Cleaned answer text (no markers)",
  "source": "dataset|ai_knowledge|news_search|unable_to_determine",
  "confidence": float
}
Performance Metrics

Average Response Time: 5-8 seconds (down from 58 seconds)
Cost per 1,000 Questions: $1-2 (down from $42)
Similarity Threshold: 85% for dataset matching
Confidence Threshold: 70% for AI and news sources
Dataset Size: 40,000+ validated questions

Technology Stack

API Framework: FastAPI
Vector Database: OpenSearch (migrated from Weaviate)
AI Models:

Gemini 2.5 Flash Lite (general questions)
GPT-4o Mini (math questions)
GPT-4 (web search)


Optimization: Parallel processing, smart model routing

Use Cases

Bangladesh Civil Service (BCS) exam preparation
Educational content validation
Bengali language question verification
Grammar and current affairs question checking
Mathematical problem validation

Key Features
Smart Model Routing
Different question types are automatically routed to optimal models:

Mathematics → GPT-4o Mini
General Knowledge → Gemini 2.5 Flash Lite
Current Affairs → Web search with GPT-4

Multi-Tier Validation
Fallback system ensures maximum coverage:

Fast dataset lookup (most common)
AI knowledge base (general questions)
Web search (current events)
Graceful fallback (unknown cases)

Bengali Language Support
Specialized handling for:

Bengali grammar questions
Unicode normalization
Bengali-specific option markers (ক), খ), গ), ঘ))

Installation
bash# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Configuration

Required environment variables:
```
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
OPENSEARCH_HOST=your_opensearch_host
OPENSEARCH_PORT=9200
API Usage
pythonimport requests

# Validate a question
response = requests.post('http://localhost:8000/validate', json={
    "question": "বাংলাদেশের রাজধানী কোথায়?",
    "options": ["ঢাকা", "চট্টগ্রাম", "খুলনা", "রাজশাহী"],
    "given_answer": "ঢাকা",
    "explanation": "বাংলাদেশের রাজধানী ঢাকা।"
})

result = response.json()
print(f"Answer Valid: {result['given_answer_valid']}")
print(f"Correct Answer: {result['final_answer']}")
License
[Your License Here]
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.Claude is AI and can make mistakes. Please double-check responses.
