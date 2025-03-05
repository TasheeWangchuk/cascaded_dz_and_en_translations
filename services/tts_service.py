# # services/tts_service.py
# import os
# import torch
# import torchaudio
# from transformers import VitsModel, AutoTokenizer, AutoProcessor
# import numpy as np
# from config import MODEL_DIR

# class TTSService:
#     def __init__(self, language):
#         self.language = language
#         self.model_path = os.path.join(MODEL_DIR, 'tts', language)
        
#         # Set device
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
#         # Load model and processor based on language
#         if language == 'en':
#             # Load MMS-TTS model for English
#             self.model = VitsModel.from_pretrained(self.model_path).to(self.device)
#             self.processor = AutoProcessor.from_pretrained(self.model_path)
#         elif language == 'dz':
#             # Load custom Dzongkha VITS model
#             self.processor = AutoProcessor.from_pretrained(self.model_path)
#             self.model = VitsModel.from_pretrained(self.model_path).to(self.device)
        
#         # Set model to evaluation mode
#         self.model.eval()
#         print(f"Loaded TTS model for {language} on {self.device}")
    
#     def synthesize(self, text, output_path):
#         """Convert text to speech and save to output_path"""
#         # Prepare the input
#         inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        
#         # Generate speech
#         with torch.no_grad():
#             speech = self.model.generate(input_ids=inputs["input_ids"]).audio[0]
        
#         # Convert to numpy and save as wav
#         speech_np = speech.cpu().numpy()
        
#         # Save the audio (16kHz is standard for these models)
#         sample_rate = 16000
#         torchaudio.save(
#             output_path,
#             torch.tensor(speech_np).unsqueeze(0),
#             sample_rate
#         )
        
#         return output_path

# services/tts_service.py
import os
import torch
import torchaudio
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import numpy as np
from config import MODEL_DIR

class TTSService:
    def __init__(self, language):
        self.language = language
        self.model_path = os.path.join(MODEL_DIR, 'tts', language)
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model and processor based on language
        if language == 'en':
            # Load MMS-TTS model for English (facebook/mms_tts_english)
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        elif language == 'dz':
            # Load custom Dzongkha VITS model
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        print(f"Loaded TTS model for {language} on {self.device}")
    
    def synthesize(self, text, output_path):
        """Convert text to speech and save to output_path"""
        # Prepare the input
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        
        # Generate speech - calling the appropriate method based on model type
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                # For VITS models
                outputs = self.model(**inputs)
                if hasattr(outputs, 'waveform'):
                    speech = outputs.waveform[0]
                else:
                    # If the model returns audio directly
                    speech = outputs[0] if isinstance(outputs, tuple) else outputs
            
        # Convert to numpy and save as wav
        speech_np = speech.cpu().numpy()
        
        # Save the audio (16kHz is standard for these models)
        sample_rate = 16000
        torchaudio.save(
            output_path,
            torch.tensor(speech_np).unsqueeze(0),
            sample_rate
        )
        
        return output_path