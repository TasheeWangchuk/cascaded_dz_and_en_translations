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