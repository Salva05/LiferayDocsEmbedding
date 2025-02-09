"""
Centralized Configuration Module

This module loads environment variables from a .env file and sets up
the centralized configuration for the application. It includes:
  - Data paths (DATA_PATH)
  - Directory for persisting the Chroma database (CHROMA_DB_DIR) (default is root project folder)
  - Logging configuration: logs are written to both a file (run.log) and the console.

Ensure that a .env file is present with the required variables.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")  # Directory to persist the Chroma DB

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

# Stream handler to output logs to the terminal as well
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
