
import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
from config import settings
from vector_db import get_vector_db, EmbeddingService

class DataPreprocessor:
    """Preprocesses dataset and creates embeddings"""
    
    def __init__(self):
        print("Initializing EmbeddingService...")
        self.embedding_service = EmbeddingService()
        print("Initializing VectorDB...")
        self.vector_db = get_vector_db()
    
    def load_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSON file"""
        print(f"Loading dataset from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ Loaded {len(data)} questions")
        return data
    
    def prepare_documents(self, data: List[Dict[str, Any]]) -> tuple:
        """Prepare documents for embedding - FIXED: Only embed questions"""
        documents = []
        texts = []
        
        for item in data:
            question_text = item.get('question', '').strip()
            
            if not question_text:
                print(f"  Warning: Skipping item {item.get('id')} - empty question")
                continue
            
            question_text = question_text.replace('\x00', '')
            question_text = ' '.join(question_text.split())
            
            document = {
                "id": item.get('id'),
                "question": question_text,
                "answer": item.get('answer'),
                "explanation": item.get('explain', ''),
                "options": json.dumps({
                    "option1": item.get('option1', ''),
                    "option2": item.get('option2', ''),
                    "option3": item.get('option3', ''),
                    "option4": item.get('option4', ''),
                    "option5": item.get('option5', '')
                }, ensure_ascii=False)
            }
            
            documents.append(document)
            texts.append(question_text)
        
        return documents, texts
    
    def create_embeddings_and_store(self, collection_name: str = "fact_check_questions"):
        """Main function to process data and create embeddings"""
        
        print("="*70)
        print("DATA PREPROCESSING FOR FACT CHECKER")
        print("="*70)
        print(f"Dataset: {settings.dataset_path}")
        print(f"Vector DB: {settings.vector_db}")
        print(f"Embedding: {settings.embedding_type}")
        print("="*70)
        
        data = self.load_dataset(settings.dataset_path)
        
        print("\nPreparing documents...")
        documents, texts = self.prepare_documents(data)
        print(f"✓ Prepared {len(documents)} documents")
        print(f"✓ Using ONLY questions for embeddings (not options/explanations)")
        
        print("\nGenerating embeddings...")
        
        if settings.embedding_type == "gemini":
            batch_size = 100
            print(f"Using Gemini batch size: {batch_size}")
        elif settings.embedding_type == "openai":
            batch_size = 100
            print(f"Using OpenAI batch size: {batch_size}")
        else:
            batch_size = 1000
            print(f"Using local GPU batch size: {batch_size}")
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_service.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        print(f"✓ Generated {len(all_embeddings)} embeddings")
        
        dimension = len(all_embeddings[0])
        print(f"\nCreating vector database collection (dimension: {dimension})...")
        self.vector_db.create_collection(collection_name, dimension)
        
        print("Storing embeddings in vector database...")
        self.vector_db.add_documents(collection_name, documents, all_embeddings)
        
        print("\n" + "="*60)
        print("VERIFYING STORAGE")
        print("="*60)
        
        print("Testing retrieval with sample question...")
        sample_question = documents[0]['question']
        print(f"Sample question: {sample_question[:100]}...")
        
        sample_embedding = self.embedding_service.embed_query(sample_question)
        results = self.vector_db.search(collection_name, sample_embedding, top_k=3)
        
        print(f"✓ Retrieved {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Question: {result['question'][:100]}...")
            print(f"  Score: {result['score']}")
            if result['question'] == sample_question:
                print(f"  ✓ EXACT MATCH FOUND!")
        
        print("\n✓ Storage verification successful!")
        print("="*60)
        
        self.vector_db.close()
        
        print("\n" + "="*70)
        print("✓ DATA PREPROCESSING COMPLETE!")
        print("="*70)
        
        return len(documents)

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("STARTING DATA PREPROCESSING")
    print("="*70 + "\n")
    
    try:
        preprocessor = DataPreprocessor()
        count = preprocessor.create_embeddings_and_store()
        
        print("\n" + "="*70)
        print(f"✓ Successfully processed {count} questions!")
        print("="*70)
        print("\nNext steps:")
        print("1. Start the backend server: python backend.py")
        print("2. Test the API: http://localhost:7001/docs")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"Please update the DATASET_PATH in your .env file")
        print(f"Current path: {settings.dataset_path}")
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()