
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    
    # Google Gemini
    google_api_key: Optional[str] = None
    
    # News API
    news_api_key: Optional[str] = None
    
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
    
    # LLM Configuration
    llm_provider: str = "gemini"  # "openai" or "gemini"
    
    # Embedding Configuration
    embedding_type: str = "openai"  # "openai" or "gemini"
    embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "models/text-embedding-004"
    local_embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    use_gpu: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()