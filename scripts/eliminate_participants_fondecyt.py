"""
Run this script to eliminate the least active participants from every audio in the database."""

import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from config import base

base_fondecyt = os.path.join(base, "fondecyt", "base")

# scan base_ceatram folder
for task in os.listdir(base_fondecyt):
    print("now processeing: ", task)
    base_fondecyt_task = os.path.join(base_fondecyt, task)
    for participante in tqdm(os.listdir(base_fondecyt_task)):
        for tipo_de_tarea in os.listdir(os.path.join(base_fondecyt_task, participante)):
            for diarization_label in os.listdir(os.path.join(base_fondecyt_task, participante, tipo_de_tarea, "diarization_labels")):
                speaker_lines = defaultdict(int)
                speech_part = []
                with open(os.path.join(base_fondecyt_task, participante, tipo_de_tarea, "diarization_labels", diarization_label), "r", encoding="utf-8") as file:
                    for line in file:
                        start, end, speaker = line.strip().split('\t')
                        speaker_lines[speaker] += 1
                        speech_part.append([start, end, speaker])

                        # Find the speaker with the majority of lines
                        majority_speaker = max(speaker_lines, key=speaker_lines.get)

                        # # Calculate the percentage of lines spoken by the majority speaker
                        # total_lines = sum(speaker_lines.values())
                        # percentage_majority = (speaker_lines[majority_speaker] / total_lines) * 100

                        # # Print the results
                        # print(f"The majority speaker is {majority_speaker} with {percentage_majority:.2f}% of lines.")
                    silence_part = []
                    for line in speech_part:
                        start, end, speaker = line
                        if majority_speaker != speaker:
                            silence_part.append([float(start), float(end)])

                    # Eliminate the least active participants
                    path_to_audio = os.path.join(base_fondecyt_task, participante, tipo_de_tarea, "mono_16khz", "loudnorm", "denoised","vad",
                                                diarization_label.replace(".wav_labels.txt", "_mono_16kHz_loudnorm_denoised_vad.wav"))

                    if not os.path.exists(os.path.join(base_fondecyt_task, participante, tipo_de_tarea, "mono_16khz", "loudnorm", "denoised","vad", "diarized")):
                        os.mkdir(os.path.join(base_fondecyt_task, participante, tipo_de_tarea, "mono_16khz", "loudnorm", "denoised","vad", "diarized"))
                    path_to_output = os.path.join(base_fondecyt_task, participante, tipo_de_tarea, "mono_16khz", "loudnorm", "denoised","vad","diarized",
                                                diarization_label.replace(".wav_labels.txt", "_mono_16kHz_loudnorm_denoised_vad_diarized.wav"))
                    # read audio with soundfile
                    audio, sr = sf.read(path_to_audio)

                    # Generate arrays for output audios
                    silence = np.ones(len(audio))
                    for intervention in silence_part:
                        silence[int(intervention[0] * sr):int(intervention[1] * sr)] = 0.0

                    # Write output audios
                    sf.write(path_to_output, audio * silence, sr)
