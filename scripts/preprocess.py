import os
import subprocess
import torch
import soundfile as sf
import noisereduce as nr
import numpy as np
import pandas as pd

def normalize_channel(audio_file):
    """
    Normalize audio to channel with common characteristics
    1. Convert to mono: -c 1
    2. Reduce bitrate to 13bps: -e gsm-full-rate
    3. Resample to 16kHz: -r 16000
    4. Compress by factor of 8: -C 8
    5. Apply bandpass filter from 200Hz to 3.4kHz:sinc 200-3.4k
    6. Decode to 16 bit (to ensure compatibility with Python)
    """

    # Define input and output files
    audio_file_name = os.path.basename(audio_file)  # get audio file name only
    # Get path of folder containing audio_file
    audio_file_path = os.path.dirname(audio_file)
    # Create subdirectory of channel normalized audio
    normalized_audio_folder = os.path.join(audio_file_path, '16000_channelnorm')
    # Create folder if it doesn't exist
    if not os.path.exists(normalized_audio_folder):
        os.makedirs(normalized_audio_folder)
    # Define output file
    base, _ = os.path.splitext(audio_file_name)
    input_1 = str(audio_file)
    output_1_name = base + '_16000_channelnorm_temp.wav'
    output_1 = os.path.join(normalized_audio_folder, output_1_name)
    # Define commands for audio channel normalisation
    command_1 = f'sox {input_1} -c 1 -e gsm-full-rate -r 16000 -C 8 {output_1} sinc 200-3.4k'
    input_2 = str(output_1)
    normalized_channel_audio_file = output_1.replace(
        '16000_channelnorm_temp.wav', '16000_channelnorm.wav')
    command_2 = f'sox {input_2} -b 16 {normalized_channel_audio_file}'
    try:
        os.system(command_1)
        os.system(command_2)
        print('Channel normalized: ', normalized_channel_audio_file)
    except Exception:
        print('Error normalizing channel')
        normalized_channel_audio_file = audio_file
    return normalized_channel_audio_file

def convert_wav_mono_16khz(audio_file):
    mono_16khz = os.path.join(
        os.path.dirname(audio_file), 'mono_16khz')
    # Create folder if it doesn't exist
    if not os.path.exists(mono_16khz):
        os.makedirs(mono_16khz)
    output_file = os.path.join(mono_16khz,
                                audio_file.replace('.wav', '_mono_16kHz.wav').split("/")[-1])
    subprocess.call(["ffmpeg", "-i", audio_file, "-vn", "-acodec",
                      "pcm_s16le", "-ar", "16000", "-ac", "1", "-strict", "-2",output_file])
    return output_file

def normalize_loudness(audio_file):
    """Loudness normalization with FFMPEG.
       Parameters:
            I: integrated loudness [dB LUFS] : -16
            LRA: loudness range [dB LUFS] : 11
            TP: max true peak [dB LUFS] : -1.5
    """
    # Create subdirectory of loudness normalized audio
    normalized_audio_folder = os.path.join(
        os.path.dirname(audio_file), 'loudnorm')
    # Create folder if it doesn't exist
    if not os.path.exists(normalized_audio_folder):
        os.makedirs(normalized_audio_folder)
    # Define output file
    base, _ = os.path.splitext(os.path.basename(audio_file))
    normalized_loudness_audio_file = os.path.join(
        normalized_audio_folder, base + '_loudnorm.wav')
    command = f'ffmpeg -hide_banner -i {audio_file} -af loudnorm=I=-16:LRA=11:TP=-1.5,aformat=sample_rates=16000 {normalized_loudness_audio_file}'
    try:
        os.system(command)
        print('Loudness normalized: ', normalized_loudness_audio_file)
    except Exception:
        print('Error normalizing loudness')
        normalized_loudness_audio_file = audio_file
    return normalized_loudness_audio_file

def denoise(audio_file):
    """stationary noise reduction with noisereduce."""
    # Create subdirectory of denoised audio
    denoised_audio_folder = os.path.join(os.path.dirname(audio_file),
                                         'denoised')
    # Create folder if it doesn't exist
    if not os.path.exists(denoised_audio_folder):
        os.makedirs(denoised_audio_folder)

    # Read audio file
    audio, sr = sf.read(audio_file)

    # n_std_thresh: Number of st devs above mean to place the threshold between signal and noise.
    reduced_noise = nr.reduce_noise(
        y=audio, sr=sr, n_std_thresh_stationary=1.5, stationary=True)

    # Define output file
    base, _ = os.path.splitext(os.path.basename(audio_file))
    denoised_audio_file = os.path.join(denoised_audio_folder, base + '_denoised.wav')
    sf.write(denoised_audio_file, reduced_noise, sr)
    return denoised_audio_file

def vad(audio_file):
    """Remove silences from audio file from VAD timestamps."""
    # Create subdirectory of VADded audio
    vad_audio_folder = os.path.join(os.path.dirname(audio_file),'vad')

    # Create folder if it doesn't exist
    if not os.path.exists(vad_audio_folder):
        os.makedirs(vad_audio_folder)

    # Load model
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = utils

    # Define VAD parameters
    threshold = 0.5
    min_speech_duration_ms = 250
    min_silence_duration_ms = 100
    sampling_rate = 16000

    wav = read_audio(audio_file, sampling_rate=sampling_rate)
    speech_timestamps = pd.DataFrame(get_speech_timestamps(wav, model, sampling_rate=sampling_rate, threshold=threshold,
                                                           min_speech_duration_ms=min_speech_duration_ms,
                                                           min_silence_duration_ms=min_silence_duration_ms))

    # Generate array for output audio
    speech = np.zeros(len(wav))

    for _, row in speech_timestamps.iterrows():
        start = row.start
        end = row.end
        speech[start:end] = 1

    vad_audio = wav * speech

    # Define output file
    base, _ = os.path.splitext(os.path.basename(audio_file))
    vad_audio_file = os.path.join(vad_audio_folder, base + '_vad.wav')
    sf.write(vad_audio_file, vad_audio, sampling_rate)
    print('VADded: ', vad_audio_file)
    return vad_audio_file

def preprocess_audio(audio_file, normalize_channel_config=False,convert_to_mono_16khz_config=True,
                      normalize_loudness_config=True, denoise_config=True, vad_config=False):
    """Audio preprocessing pipeline."""
    if normalize_channel_config:
        audio_file = normalize_channel(audio_file)
    if convert_to_mono_16khz_config:
        audio_file = convert_wav_mono_16khz(audio_file)
    if normalize_loudness_config:
        audio_file = normalize_loudness(audio_file)
    if denoise_config:
        audio_file = denoise(audio_file)
    if vad_config:
        audio_file = vad(audio_file)
    return audio_file
