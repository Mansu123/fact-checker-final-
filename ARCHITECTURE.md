

# System Architecture

## Overview

The Fact Checking & Question Generation System is designed as a modular, scalable application that combines vector search, LLM reasoning, and a modern web interface.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                          │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    React Application                      │  │
│  │  ┌──────────────┐      ┌──────────────────────────┐      │  │
│  │  │ Fact Check   │      │  Question Generation     │      │  │
│  │  │   Component  │      │      Component           │      │  │
│  │  └──────────────┘      └──────────────────────────┘      │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           │ HTTP/REST API
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                      Backend Layer (FastAPI)                    │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │   Fact       │  │   Question   │  │   API Routes         │ │
│  │   Checker    │  │   Generator  │  │   & Controllers      │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────┘ │
│         │                  │                                    │
└─────────┼──────────────────┼────────────────────────────────────┘
          │                  │
          ├──────────────────┴────────────────┐
          │                                   │
┌─────────┴──────────┐              ┌────────┴───────────┐
│   LLM Service      │              │  Vector Search      │
│   (OpenAI GPT-4)   │              │    Service          │
│                    │              │                     │
│  ┌──────────────┐  │              │  ┌──────────────┐  │
│  │ Verification │  │              │  │  Embedding   │  │
│  │   Prompts    │  │              │  │  Generator   │  │
│  └──────────────┘  │              │  └──────────────┘  │
│  ┌──────────────┐  │              │                     │
│  │ Generation   │  │              │  ┌──────────────┐  │
│  │   Prompts    │  │              │  │   Retriever  │  │
│  └──────────────┘  │              │  └──────────────┘  │
└────────────────────┘              └────────┬───────────┘
                                             │
                    ┌────────────────────────┴────────────────────┐
                    │         Vector Database Layer               │
                    │                                             │
                    │  ┌──────────┐  ┌──────┐  ┌────────────┐   │
                    │  │ Weaviate │  │ FAISS│  │   Milvus   │   │
                    │  └──────────┘  └──────┘  └────────────┘   │
                    │  ┌──────────┐  ┌──────────────────────┐   │
                    │  │  Qdrant  │  │     Zilliz           │   │
                    │  └──────────┘  └──────────────────────┘   │
                    └─────────────────────────────────────────────┘
```

## Component Details

### 1. Frontend Layer

**Technology**: React (Vanilla JS with Babel)

**Components**:
- **App Component**: Main container with tab navigation
- **FactCheckTab**: Interface for verifying questions and answers
- **GenerateTab**: Interface for generating new questions
- **ResultCard**: Display verification results
- **GeneratedQuestionCard**: Display generated questions

**Features**:
- Single-page application (SPA)
- Real-time API communication
- Responsive design with Tailwind CSS
- Support for 1-10 batch operations

### 2. Backend Layer

**Technology**: FastAPI (Python)

**Core Services**:

#### a) Fact Checker Service
```python
Class: FactChecker
Methods:
- verify_answer(): Single answer verification
- verify_explanation(): Explanation validation
- batch_verify(): Multiple questions (1-10)
- retrieve_similar_questions(): Vector search
```

**Process Flow**:
1. Receive question and answer
2. Generate query embedding
3. Search vector DB for similar questions
4. Build context from similar questions
5. Send to LLM with verification prompt
6. Parse and return structured result

#### b) Question Generator Service
```python
Class: QuestionGenerator
Methods:
- generate_from_topic(): Topic-based generation
- generate_similar_question(): Variation generation
- generate_random_questions(): Random generation
- generate_variations(): Multiple variations
```

**Process Flow**:
1. Receive generation request
2. Search for relevant examples
3. Build context from examples
4. Send to LLM with generation prompt
5. Parse JSON response
6. Return generated questions

#### c) API Routes
- `/fact-check/*`: Fact checking endpoints
- `/generate/*`: Question generation endpoints
- `/search`: Vector search endpoint
- `/health`: Health check endpoint

### 3. LLM Service

**Provider**: OpenAI
**Model**: GPT-4

**Usage**:

#### Verification Prompts
- Structured system prompts
- Context from vector search
- Clear output format requirements
- Confidence scoring

#### Generation Prompts
- Example-based learning
- Style matching
- Format specifications
- Language control

**Optimization**:
- Temperature control (0 for verification, 0.7 for generation)
- Token management
- Prompt caching
- Error handling

### 4. Vector Search Service

**Embedding Service**:
- Model: `text-embedding-3-small` (default)
- Dimension: 1536
- Provider: OpenAI

**Vector Databases**:

#### Weaviate (Primary)
- **Type**: Cloud-hosted
- **Strengths**: Scalability, managed service, GraphQL API
- **Use Case**: Production deployment
- **Connection**: REST API

#### FAISS (Fallback)
- **Type**: Local
- **Strengths**: Fast, simple, no server required
- **Use Case**: Development, offline usage
- **Storage**: Local files

#### Milvus
- **Type**: Local/Cloud
- **Strengths**: Feature-rich, enterprise-ready
- **Use Case**: Large-scale production
- **Connection**: gRPC

#### Qdrant
- **Type**: Local/Cloud
- **Strengths**: Modern API, filtering capabilities
- **Use Case**: Advanced search requirements
- **Connection**: HTTP/gRPC

#### Zilliz
- **Type**: Cloud (Managed Milvus)
- **Strengths**: Managed, scalable
- **Use Case**: Cloud production
- **Connection**: gRPC

### 5. Data Layer

**Input Data**:
```json
{
  "id": integer,
  "question": string,
  "option1-5": string,
  "answer": integer (1-5),
  "explain": string
}
```

**Embedding Storage**:
- Document vectors: 1536 dimensions
- Metadata: Question, options, answer index, explanation
- Search index: Optimized for similarity search

**Optional MySQL**:
- Store verification history
- Track generation requests
- Analytics and reporting

## Data Flow

### Fact Checking Flow

```
User Input → Frontend → API Request → Backend
                                        ↓
                              Generate Embedding
                                        ↓
                           Vector DB Search (Top 3)
                                        ↓
                              Build Context
                                        ↓
                         LLM Verification (GPT-4)
                                        ↓
                        Parse & Structure Result
                                        ↓
                        API Response → Frontend
                                        ↓
                            Display Result
```

### Question Generation Flow

```
User Input → Frontend → API Request → Backend
                                        ↓
                              Get Examples from DB
                                        ↓
                              Build Context
                                        ↓
                        LLM Generation (GPT-4)
                                        ↓
                        Parse JSON Response
                                        ↓
                        Validate Format
                                        ↓
                        API Response → Frontend
                                        ↓
                        Display Questions
```

### Data Preprocessing Flow

```
JSON Dataset → Load & Parse
                    ↓
           Prepare Documents
                    ↓
        Generate Embeddings (Batch)
                    ↓
        Create Vector Collection
                    ↓
        Store in Vector DB
                    ↓
            Ready for Search
```

## Scalability Considerations

### Horizontal Scaling

1. **Backend**: Multiple FastAPI instances with load balancer
2. **Vector DB**: Distributed deployment (Weaviate/Milvus clusters)
3. **LLM**: Rate limiting and queueing

### Vertical Scaling

1. **Increase batch sizes**
2. **Use GPU for embeddings** (with `faiss-gpu`)
3. **Optimize vector index** (IVF, HNSW)

### Performance Optimization

1. **Caching**:
   - Embedding cache for repeated queries
   - LLM response cache for similar requests
   
2. **Batch Processing**:
   - Group embedding generation
   - Batch LLM requests
   
3. **Index Optimization**:
   - Appropriate index type for data size
   - Regular index rebuilding

## Security Considerations

1. **API Security**:
   - Rate limiting
   - Input validation
   - CORS configuration
   
2. **Data Security**:
   - API key management (env variables)
   - Secure database connections
   - Input sanitization

3. **LLM Safety**:
   - Prompt injection prevention
   - Output validation
   - Content filtering

## Monitoring & Logging

**Key Metrics**:
- API response times
- LLM request latency
- Vector search performance
- Error rates
- Resource utilization

**Logging**:
- Request/response logs
- Error logs
- Performance logs
- Usage analytics

## Deployment Options

### Option 1: Local Development
- FAISS for vector storage
- Local API server
- File-based frontend

### Option 2: Cloud Production
- Weaviate Cloud
- Cloud-hosted FastAPI
- CDN for frontend
- MySQL for persistence

### Option 3: Docker Deployment
- All services containerized
- Docker Compose orchestration
- Milvus or Qdrant in containers

## Technology Stack Summary

| Layer | Technology | Version |
|-------|-----------|---------|
| Frontend | React | 18 |
| UI Framework | Tailwind CSS | 3.x |
| Backend | FastAPI | 0.104+ |
| LLM Framework | LangChain | 0.1+ |
| LLM Provider | OpenAI | GPT-4 |
| Embedding | OpenAI | ada-002/003 |
| Vector DB | Weaviate/FAISS/Milvus/Qdrant | Latest |
| Database | MySQL (Optional) | 8.0 |
| Python | 3.8+ | - |

## Future Enhancements

1. **User Authentication**: Add user accounts and history
2. **Advanced Analytics**: Dashboard for insights
3. **Multi-model Support**: Support other LLMs (Claude, Gemini)
4. **Real-time Collaboration**: Multiple users editing simultaneously
5. **Mobile App**: Native mobile applications
6. **API Gateway**: Advanced routing and caching
7. **Kubernetes Deployment**: Container orchestration
8. **Enhanced Caching**: Redis for distributed caching