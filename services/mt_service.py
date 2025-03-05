# services/mt_service.py
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,NllbTokenizer
from safetensors.torch import load_file
from config import MODEL_DIR

class MTService:


    def __init__(self, direction):
        self.direction = direction
        if direction == 'en2dz' or direction == 'dz2en':
            self.model_path = os.path.join(MODEL_DIR, 'mt', direction)
            self.tokenizer_path = os.path.join(self.model_path)

            print(self.model_path )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            
            # OPTION 1: Load directly from the checkpoint folder if config.json contains architecture
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            
            # OPTION 2: If you must load separately, use strict=False to ignore missing keys
            # self.model = AutoModelForSeq2SeqLM.from_pretrained(self.tokenizer_path)

            # model_safetensors = os.path.join(self.model_path, 'model.safetensors')
            # state_dict = load_file(model_safetensors)
            # self.model.load_state_dict(state_dict, strict=False)
            
            # Move model to GPU if available
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"Loaded MT model for {direction} on {self.device}")
        
    def translate(self, text):
        """Translate text from source language to target language"""
        # Prepare the model inputs
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate translation
        with torch.no_grad():
            translated_ids = self.model.generate(**inputs, max_length=512)
        
        # Decode the output
        translation = self.tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        
        return translation