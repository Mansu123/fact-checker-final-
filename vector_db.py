import json
import os
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
import time

# OpenSearch imports
from opensearchpy import OpenSearch, helpers
from config import settings

class VectorDBInterface(ABC):
    """Abstract interface for vector databases"""
    
    @abstractmethod
    def create_collection(self, collection_name: str, dimension: int):
        pass
    
    @abstractmethod
    def add_documents(self, collection_name: str, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        pass
    
    @abstractmethod
    def search(self, collection_name: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def close(self):
        pass

class OpenSearchDB(VectorDBInterface):
    """OpenSearch vector database implementation with kNN support"""
    
    def __init__(self):
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                conn_params = {
                    'hosts': [{'host': settings.opensearch_host, 'port': settings.opensearch_port}],
                    'http_compress': True,
                    'use_ssl': settings.opensearch_use_ssl,
                    'verify_certs': False,
                    'ssl_assert_hostname': False,
                    'ssl_show_warn': False,
                    'timeout': 30,
                    'max_retries': 3,
                    'retry_on_timeout': True
                }
                
                if settings.opensearch_use_ssl:
                    conn_params['http_auth'] = (settings.opensearch_user, settings.opensearch_password)
                
                self.client = OpenSearch(**conn_params)
                
                if self.client.ping():
                    print("✓ Connected to OpenSearch")
                    info = self.client.info()
                    print(f"✓ OpenSearch version: {info['version']['number']}")
                    health = self.client.cluster.health()
                    print(f"✓ Cluster status: {health['status']}")
                    return
                else:
                    print(f"⚠ Attempt {attempt + 1}/{max_retries}: OpenSearch ping failed")
                
            except Exception as e:
                print(f"⚠ Attempt {attempt + 1}/{max_retries}: Connection failed - {e}")
            
            if attempt < max_retries - 1:
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        raise Exception("OpenSearch is not ready")
    
    def create_collection(self, collection_name: str, dimension: int):
        try:
            if self.client.indices.exists(index=collection_name):
                self.client.indices.delete(index=collection_name)
                print(f"✓ Deleted existing index: {collection_name}")
            
            index_body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100,
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    }
                },
                "mappings": {
                    "properties": {
                        "question_id": {"type": "integer"},
                        "question": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "answer": {"type": "integer"},
                        "explanation": {"type": "text"},
                        "options": {"type": "text"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "l2",
                                "engine": "nmslib",
                                "parameters": {"ef_construction": 128, "m": 24}
                            }
                        }
                    }
                }
            }
            
            self.client.indices.create(index=collection_name, body=index_body)
            print(f"✓ Created OpenSearch index: {collection_name} with dimension: {dimension}")
            
        except Exception as e:
            print(f"✗ Error creating index: {e}")
            raise
    
    def add_documents(self, collection_name: str, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        try:
            actions = []
            
            for i, (doc, emb) in enumerate(zip(documents, embeddings)):
                action = {
                    "_index": collection_name,
                    "_id": str(doc.get("id", i)),
                    "_source": {
                        "question_id": doc.get("id"),
                        "question": doc.get("question"),
                        "answer": doc.get("answer"),
                        "explanation": doc.get("explanation"),
                        "options": doc.get("options"),
                        "embedding": emb
                    }
                }
                actions.append(action)
                
                if len(actions) >= 1000:
                    helpers.bulk(self.client, actions)
                    print(f"✓ Added {i + 1}/{len(documents)} documents")
                    actions = []
            
            if actions:
                helpers.bulk(self.client, actions)
            
            self.client.indices.refresh(index=collection_name)
            print(f"✓ Added all {len(documents)} documents to OpenSearch")
            
        except Exception as e:
            print(f"✗ Error adding documents: {e}")
            raise
    
    def search(self, collection_name: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            query = {
                "size": top_k,
                "query": {"knn": {"embedding": {"vector": query_embedding, "k": top_k}}},
                "_source": ["question_id", "question", "answer", "explanation", "options"]
            }
            
            response = self.client.search(index=collection_name, body=query)
            
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                results.append({
                    "id": source.get("question_id"),
                    "question": source.get("question"),
                    "answer": source.get("answer"),
                    "explanation": source.get("explanation"),
                    "options": source.get("options"),
                    "score": hit['_score']
                })
            
            return results
            
        except Exception as e:
            print(f"✗ Error searching: {e}")
            return []
    
    def close(self):
        if hasattr(self, 'client'):
            self.client.close()
            print("✓ Closed OpenSearch connection")

def get_vector_db() -> VectorDBInterface:
    return OpenSearchDB()

class DirectOpenAIEmbeddings:
    """Direct OpenAI API embeddings without langchain"""
    
    def __init__(self, model: str, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        print(f"✓ Using OpenAI embeddings: {model}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            cleaned_batch = []
            for text in batch:
                if text and isinstance(text, str) and text.strip():
                    cleaned_text = text.strip()
                    cleaned_text = cleaned_text.replace('\x00', '')
                    cleaned_batch.append(cleaned_text)
                else:
                    cleaned_batch.append("placeholder text")
            
            try:
                response = self.client.embeddings.create(input=cleaned_batch, model=self.model)
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"\n✗ Error in batch {i//batch_size + 1}: {e}")
                print(f"  Batch size: {len(cleaned_batch)}")
                print(f"  Processing items individually...")
                
                for j, text in enumerate(cleaned_batch):
                    try:
                        response = self.client.embeddings.create(input=[text], model=self.model)
                        embeddings.append(response.data[0].embedding)
                    except:
                        print(f"  Skipping item {i+j}: {text[:50]}...")
                        embeddings.append([0.0] * 1536)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        if not text or not isinstance(text, str):
            text = "placeholder"
        text = text.strip().replace('\x00', '')
        
        response = self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding

class DirectGeminiEmbeddings:
    """Direct Gemini API embeddings"""
    
    def __init__(self, model: str, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = model
        print(f"✓ Using Gemini embeddings: {model}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import google.generativeai as genai
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            cleaned_batch = []
            for text in batch:
                if text and isinstance(text, str) and text.strip():
                    cleaned_text = text.strip()
                    cleaned_text = cleaned_text.replace('\x00', '')
                    cleaned_batch.append(cleaned_text)
                else:
                    cleaned_batch.append("placeholder text")
            
            try:
                for text in cleaned_batch:
                    result = genai.embed_content(
                        model=self.model,
                        content=text,
                        task_type="retrieval_document"
                    )
                    embeddings.append(result['embedding'])
            except Exception as e:
                print(f"\n✗ Error in batch {i//batch_size + 1}: {e}")
                print(f"  Batch size: {len(cleaned_batch)}")
                print(f"  Processing items individually...")
                
                for j, text in enumerate(cleaned_batch):
                    try:
                        result = genai.embed_content(
                            model=self.model,
                            content=text,
                            task_type="retrieval_document"
                        )
                        embeddings.append(result['embedding'])
                    except:
                        print(f"  Skipping item {i+j}: {text[:50]}...")
                        embeddings.append([0.0] * 768)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        import google.generativeai as genai
        
        if not text or not isinstance(text, str):
            text = "placeholder"
        text = text.strip().replace('\x00', '')
        
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']

class EmbeddingService:
    """Service for generating embeddings"""
    
    def __init__(self):
        self.embedding_type = settings.embedding_type
        
        if self.embedding_type == "gemini":
            if not settings.google_api_key:
                raise Exception("GOOGLE_API_KEY not found in .env file")
            self.embeddings = DirectGeminiEmbeddings(
                model=settings.gemini_embedding_model,
                api_key=settings.google_api_key
            )
        elif self.embedding_type == "openai":
            self.embeddings = DirectOpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.openai_api_key
            )
        else:
            raise Exception("Invalid embedding_type. Use 'openai' or 'gemini'")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)