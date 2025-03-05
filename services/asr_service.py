# # services/asr_service.py
# import os
# import torch
# import torchaudio
# from transformers import WhisperProcessor, WhisperForConditionalGeneration,  Wav2Vec2ForCTC, Wav2Vec2Processor
# from safetensors.torch import load_file
# from config import MODEL_DIR

# class ASRService:
#     def __init__(self, language):
#         self.language = language
        
#         if language == 'en':
#             # Load Whisper-small for English
#             self.model_path = os.path.join(MODEL_DIR, 'asr', 'en')
#             self.processor = WhisperProcessor.from_pretrained(self.model_path)
#             self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
#             self.model.eval()  # Set to evaluation mode
#         elif language == 'dz':
#             # Load custom Dzongkha ASR model from safetensors
#             self.model_path = os.path.join(MODEL_DIR, 'asr', 'dz')
#             self.tokenizer_path = os.path.join(self.model_path)
#             self.processor = Wav2Vec2Processor.from_pretrained(self.tokenizer_path)
            
#             # Load model weights from safetensors
#             self.model = Wav2Vec2ForCTC.from_pretrained(self.tokenizer_path)
#             model_safetensors = os.path.join(self.model_path, 'model.safetensors')
#             state_dict = load_file(model_safetensors)
#             self.model.load_state_dict(state_dict)
#             self.model.eval()
        
#         # Move model to GPU if available
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.model = self.model.to(self.device)
#         print(f"Loaded ASR model for {language} on {self.device}")
        
#     def transcribe(self, audio_path):
#         """Transcribe speech from audio file to text"""
#         # Load and preprocess the audio
#         audio, sample_rate = torchaudio.load(audio_path)
        
#         # Resample if needed
#         if sample_rate != 16000:
#             resampler = torchaudio.transforms.Resample(sample_rate, 16000)
#             audio = resampler(audio)
#             sample_rate = 16000
        
#         # Convert to mono if needed
#         if audio.shape[0] > 1:
#             audio = torch.mean(audio, dim=0, keepdim=True)
        
#         # Process audio with the model
#         input_features = self.processor(
#             audio.squeeze().numpy(), 
#             sampling_rate=sample_rate, 
#             return_tensors="pt"
#         ).input_features.to(self.device)
        
#         # Generate transcription
#         with torch.no_grad():
#             if self.language == 'en':
#                 forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="en", task="transcribe")
#                 generated_ids = self.model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
#             else:  # Dzongkha
#                 generated_ids = self.model.generate(input_features)
        
#         # Decode the transcription
#         transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
#         return transcription

# services/asr_service.py
import os
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2ForCTC, Wav2Vec2Processor
from config import MODEL_DIR

class ASRService:
    def __init__(self, language):
        self.language = language
        self.model_path = os.path.join(MODEL_DIR, 'asr', language)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if language == 'en':
            # Load Whisper-small for English
            self.processor = WhisperProcessor.from_pretrained(self.model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
        elif language == 'dz':
            # Load Dzongkha ASR model
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Loaded ASR model for {language} on {self.device}")
    
    def transcribe(self, audio_path):
        """Transcribe speech from audio file to text"""
        # Load and preprocess the audio
        audio, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio = resampler(audio)
            sample_rate = 16000
        
        # Convert to mono if needed
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Process audio differently based on model type
        audio_numpy = audio.squeeze().numpy()
        
        with torch.no_grad():
            if self.language == 'en':
                # Whisper processing for English
                inputs = self.processor(
                    audio_numpy,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).to(self.device)
                
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="en", task="transcribe")
                generated_ids = self.model.generate(**inputs, forced_decoder_ids=forced_decoder_ids)
                transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            else:  # Dzongkha with Wav2Vec2
                inputs = self.processor(
                    audio_numpy,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).to(self.device)
                
                logits = self.model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
                
        return transcription