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