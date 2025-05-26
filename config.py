# app/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Config:
    # Flask Settings
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    PORT = int(os.getenv("FLASK_PORT", 5001))  # Default: 5001
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///database.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-key")  # Default weak key for development

    # Botb Scraper & preprocessing
    IMAGES_FOLDER = os.getenv("IMAGES_FOLDER", "botb_data/comp-images")
    TARGET_SIZE = (224, 224)
    NUM_AUGS = 1
    PROCESSED_DATA_PATH = 'botb_data/processed_data.npz'
    
    TARGET_SHAPE = (224, 224, 3)
    BATCH_SIZE = 8
    EPOCHS = 10
    LOG_DIR = 'logs'

    # API Keys (optional)
    # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    # STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")