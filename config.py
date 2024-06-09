import os

from pydantic_settings import BaseSettings
# from pydantic import BaseSettings


class Settings(BaseSettings):
    API_KEY: str
    SECRET_KEY: str

    DATA_PATH: str
    MODEL_NAME: str
    MODEL_FOLDER: str
    MODEL_LOCATION: str
    REPLAY_MEM_SIZE: int
    BATCH_SIZE: int
    GAMMA: float
    EPS_START: int
    EPS_END: float
    EPS_STEPS: int
    LEARNING_RATE: float
    INPUT_DIM: int
    HIDDEN_DIM: int
    ACTION_NUMBER: int
    TARGET_UPDATE: int
    N_TEST: int
    TRADING_PERIOD: int
    DOUBLE: bool
    NUMBER_EPOCHS: int

    class Config:
        # env_file = '/.env'
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        # env_file = dirname(abspath(__file__)) + '/.env'
        print("Using config file:", env_file)


settings = Settings()
