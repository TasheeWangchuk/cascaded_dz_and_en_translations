# utils/file_handling.py
import os
import uuid
from werkzeug.utils import secure_filename

def save_upload(file_obj, upload_folder):
    """Save an uploaded file to disk with a secure filename"""
    filename = secure_filename(file_obj.filename)
    # Add UUID to ensure uniqueness
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(upload_folder, unique_filename)
    file_obj.save(file_path)
    return file_path

def cleanup_file(file_path):
    """Remove a temporary file if it exists"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
    except Exception as e:
        print(f"Error cleaning up file {file_path}: {str(e)}")
    return False

# config.py
import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
TEMP_FOLDER = os.path.join(BASE_DIR, 'temp')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')

# Allowed file extensions
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}

# Check if CUDA is available and set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Model paths - update these with your actual paths
ASR_MODELS = {
    'en': {
        'model_type': 'whisper',
        'model_size': 'small',
        'path': os.path.join(MODEL_DIR, 'asr', 'en', 'whisper-small')
    },
    'dz': {
        'model_type': 'custom',
        'path': os.path.join(MODEL_DIR, 'asr', 'dz')
    }
}

MT_MODELS = {
    'en2dz': {
        'path': os.path.join(MODEL_DIR, 'mt', 'en2dz')
    },
    'dz2en': {
        'path': os.path.join(MODEL_DIR, 'mt', 'dz2en')
    }
}

TTS_MODELS = {
    'en': {
        'model_type': 'speecht5',
        'path': os.path.join(MODEL_DIR, 'tts', 'en', 'speecht5')
    },
    'dz': {
        'model_type': 'custom',
        'path': os.path.join(MODEL_DIR, 'tts', 'dz')
    }
}