import os
from tqdm import tqdm
from preprocess import preprocess_audio

# Path to the directory containing the audio files
audio_dir = '/home/udesa_ubuntu/cnc_dataset_preprocesing/audios_fluidez_chile'

# os walk to get all the files in the directory
for root, dirs, files in os.walk(audio_dir):
    for file in tqdm(files):
        # Check if the file is an audio file
        audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"]

        if any(file.endswith(ext) for ext in audio_extensions):
            # Preprocess the audio file
            preprocess_audio(os.path.join(root, file), 'data/fondecyt/preprocessed_audio')
