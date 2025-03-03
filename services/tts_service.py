# services/tts_service.py
import os
import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from safetensors.torch import load_file
import numpy as np
from config import MODEL_DIR

class TTSService:
    def __init__(self, language):
        self.language = language
        
        if language == 'en':
            # Load SpeechT5 model for English
            self.model_path = os.path.join(MODEL_DIR, 'tts', 'en', 'speecht5')
            self.processor = SpeechT5Processor.from_pretrained(self.model_path)
            self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_path)
            self.vocoder = SpeechT5HifiGan.from_pretrained(self.model_path)
            
            # Load speaker embedding
            self.speaker_embeddings_path = os.path.join(self.model_path, 'speaker_embeddings.pt')
            self.speaker_embeddings = torch.load(self.speaker_embeddings_path)
            
        elif language == 'dz':
            # Load custom Dzongkha TTS model
            self.model_path = os.path.join(MODEL_DIR, 'tts', 'dz')
            self.tokenizer_path = os.path.join(self.model_path, 'tokenizer')
            
            self.processor = SpeechT5Processor.from_pretrained(self.tokenizer_path)
            
            # Load model from safetensors
            self.model = SpeechT5ForTextToSpeech.from_pretrained(self.tokenizer_path)
            model_safetensors = os.path.join(self.model_path, 'model.safetensors')
            state_dict = load_file(model_safetensors)
            self.model.load_state_dict(state_dict)
            
            # Load vocoder from safetensors
            self.vocoder = SpeechT5HifiGan.from_pretrained(self.tokenizer_path)
            vocoder_safetensors = os.path.join(self.model_path, 'vocoder.safetensors')
            vocoder_state_dict = load_file(vocoder_safetensors)
            self.vocoder.load_state_dict(vocoder_state_dict)
            
            # Load speaker embedding for Dzongkha
            self.speaker_embeddings_path = os.path.join(self.model_path, 'speaker_embeddings.pt')
            self.speaker_embeddings = torch.load(self.speaker_embeddings_path)
        
        # Move models to GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.vocoder = self.vocoder.to(self.device)
        self.model.eval()
        self.vocoder.eval()
        
        print(f"Loaded TTS model for {language} on {self.device}")
        
    def synthesize(self, text, output_path):
        """Convert text to speech and save to output_path"""
        # Prepare the input
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        
        # Get speaker embeddings
        speaker_embeddings = self.speaker_embeddings.to(self.device)
        
        # Generate speech
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"], 
                speaker_embeddings=speaker_embeddings,
                vocoder=self.vocoder
            )
        
        # Convert to numpy and save as wav
        speech_np = speech.cpu().numpy()
        # Target sample rate for the models is typically 16kHz
        sample_rate = 16000
        
        # Save the audio
        torchaudio.save(
            output_path,
            torch.tensor(speech_np).unsqueeze(0),
            sample_rate
        )
        
        return output_path