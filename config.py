
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    
    # OpenSearch Configuration
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_user: str = "admin"
    opensearch_password: str = "admin12345678"
    opensearch_use_ssl: bool = True
    
    # Vector DB Selection
    vector_db: str = "opensearch"
    
    # MySQL
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str
    mysql_database: str = "fact_check_db"
    
    # Dataset
    dataset_path: str
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 7001
    
    # Embedding Configuration
    embedding_type: str = "local"  # "openai" or "local"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    use_gpu: bool = True  # Use M1 GPU (MPS) for local embeddings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()