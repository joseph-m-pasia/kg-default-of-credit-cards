import os

class Settings:
    ENV: str = os.getenv("ENV", "dev")  # default to "dev" if ENV variable is not set

settings = Settings()
