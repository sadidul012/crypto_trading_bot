import os

from pydantic import computed_field
from pydantic_settings import BaseSettings
# from pydantic import BaseSettings


class Settings(BaseSettings):
    API_KEY: str
    SECRET_KEY: str

    class Config:
        # env_file = '/.env'
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        # env_file = dirname(abspath(__file__)) + '/.env'
        print("Using config file:", env_file)


settings = Settings()
