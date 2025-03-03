# utils/audio_processing.py
import os
import subprocess
import torch
import torchaudio
from config import TEMP_FOLDER

def convert_audio_format(input_path, target_format='wav', sample_rate=16000):
    """
    Convert audio to a standard format for processing
    Returns path to converted file
    """
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    
    # Generate output path with new extension
    filename = os.path.basename(input_path)
    basename = os.path.splitext(filename)[0]
    output_path = os.path.join(TEMP_FOLDER, f"{basename}.{target_format}")
    
    # Simple format check - if already correct format and we want to verify sample rate
    if input_path.endswith(f".{target_format}"):
        try:
            # Check if the sample rate is already correct
            metadata = torchaudio.info(input_path)
            if metadata.sample_rate == sample_rate and metadata.num_channels == 1:
                return input_path
        except Exception:
            pass  # If there's an error, continue with conversion
    
    try:
        # Using torchaudio for audio conversion
        waveform, sr = torchaudio.load(input_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        # Save the processed audio
        torchaudio.save(output_path, waveform, sample_rate)
        return output_path
        
    except Exception as e:
        print(f"Error with torchaudio conversion: {str(e)}")
        
        # Fallback to ffmpeg
        try:
            cmd = [
                'ffmpeg', '-i', input_path, 
                '-ar', str(sample_rate), 
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file if it exists
                output_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return output_path
        except Exception as e2:
            print(f"Error with ffmpeg conversion: {str(e2)}")
            # If all conversions fail, return original file
            return input_path
