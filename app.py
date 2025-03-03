# app.py
from flask import Flask, request, jsonify, render_template, send_file
import os
import tempfile
import uuid
from services.asr_service import ASRService
from services.mt_service import MTService
from services.tts_service import TTSService
from utils.audio_processing import convert_audio_format
from utils.file_handling import save_upload, cleanup_file
from config import UPLOAD_FOLDER, ALLOWED_AUDIO_EXTENSIONS, OUTPUT_FOLDER

app = Flask(__name__)

# Initialize services - will load the actual models
print("Loading ASR models...")
asr_en = ASRService('en')
asr_dz = ASRService('dz')

print("Loading MT models...")
mt_en2dz = MTService('en2dz')
mt_dz2en = MTService('dz2en')

print("Loading TTS models...")
tts_en = TTSService('en')
tts_dz = TTSService('dz')

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

@app.route('/api/translate', methods=['POST'])
def translate_speech():
    try:
        # Get source and target languages
        source_lang = request.form.get('source_lang')
        target_lang = request.form.get('target_lang')
        
        if source_lang not in ['en', 'dz'] or target_lang not in ['en', 'dz']:
            return jsonify({'error': 'Unsupported language pair'}), 400
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        if not allowed_file(audio_file.filename):
            return jsonify({'error': f'File format not supported. Use one of: {", ".join(ALLOWED_AUDIO_EXTENSIONS)}'}), 400
        
        # Save the uploaded audio file
        audio_path = save_upload(audio_file, UPLOAD_FOLDER)
        
        # Convert to standard format for processing
        processed_audio_path = convert_audio_format(audio_path)
        
        # Step 1: Speech recognition (ASR)
        print(f"Transcribing {processed_audio_path}...")
        asr_service = asr_en if source_lang == 'en' else asr_dz
        transcription = asr_service.transcribe(processed_audio_path)
        print(f"Transcription: {transcription}")
        
        # Step 2: Machine translation (MT)
        print(f"Translating text...")
        mt_service = mt_en2dz if source_lang == 'en' else mt_dz2en
        translation = mt_service.translate(transcription)
        print(f"Translation: {translation}")
        
        # Step 3: Text-to-speech (TTS)
        print(f"Synthesizing speech...")
        tts_service = tts_dz if target_lang == 'dz' else tts_en
        output_audio_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4()}.wav")
        tts_service.synthesize(translation, output_audio_path)
        
        # Clean up temporary files
        cleanup_file(audio_path)
        if audio_path != processed_audio_path:
            cleanup_file(processed_audio_path)
        
        return jsonify({
            'success': True,
            'transcription': transcription,
            'translation': translation,
            'audio_url': f"/api/audio/{os.path.basename(output_audio_path)}"
        })
    
    except Exception as e:
        import traceback
        print(f"Error in translation process: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/audio/<filename>')
def get_audio(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True)