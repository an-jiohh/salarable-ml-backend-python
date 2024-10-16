from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    pinecone_api_key: str 
    openai_api_key: str
    pinecone_index_name: str
    pre_processed_dataset: str
    # APP_ENV: str = 'dev'
    model_config = SettingsConfigDict(env_file=".env")

@lru_cache
def get_settings():
    return Settings()