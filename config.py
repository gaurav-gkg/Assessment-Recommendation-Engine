from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GOOGLE_API_KEY: str = ""

    # paths
    DATA_DIR: Path = Path("data")
    RAW_DATA_PATH: Path = Path("data/raw/shl_catalogue.json")
    PROCESSED_DATA_PATH: Path = Path("data/processed/assessments.json")
    FAISS_INDEX_PATH: Path = Path("data/processed/faiss.index")
    FAISS_META_PATH: Path = Path("data/processed/faiss_meta.json")
    TRAIN_DATASET_PATH: Path = Path("data/datasets/train.json")
    TEST_DATASET_PATH: Path = Path("data/datasets/test.json")
    PREDICTIONS_PATH: Path = Path("data/datasets/predictions.csv")

    # scraper
    SHL_CATALOGUE_URL: str = "https://www.shl.com/solutions/products/product-catalog/"
    SHL_INDIVIDUAL_TYPE: int = 0  # 0 = Individual Test Solutions
    REQUEST_DELAY: float = 1.2
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3

    # embeddings – free-tier quota: 100 items/min (v1beta, gemini-embedding-001)
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 3072
    EMBEDDING_BATCH_SIZE: int = 50

    # LLM
    LLM_MODEL: str = "gemini-2.0-flash"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048

    # recommendations
    MIN_RECOMMENDATIONS: int = 5
    MAX_RECOMMENDATIONS: int = 10
    RETRIEVAL_K: int = 30  # FAISS candidates before reranking

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "SHL Assessment Recommendation Engine"
    API_VERSION: str = "1.0.0"
    STREAMLIT_PORT: int = 8501


settings = Settings()
