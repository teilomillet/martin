import os
from dotenv import load_dotenv

load_dotenv()
class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

    S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
    S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
    CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", None)
    CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", None)
    ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY")
    DATABASE_URL = "sqlite:///./app.db" # DATABASE_URL = "postgresql://user:password@postgresserver/db"
    LOG_FILE = "chat.log"
