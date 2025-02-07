import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
