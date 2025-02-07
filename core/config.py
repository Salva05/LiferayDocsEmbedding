import os
import logging
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Logging setup
log_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'run.log')

if not os.path.exists(log_filename):
    open(log_filename, 'w').close()  # Create log file if it doesn't exist

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger()



