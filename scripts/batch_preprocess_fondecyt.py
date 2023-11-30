import os
from tqdm import tqdm
from scripts.preprocess import preprocess_audio
from config import base

base = os.path.join(base, "fondecyt", "base") # Path to the base folder A1085
diarize = False
GPU = False
preprocess = True


if diarize:
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token="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")

    if GPU:
        import torch
        pipeline.to(torch.device("cuda"))


tareas = ["AD","bvFTD","CN_1", "CN_2", "PD"]

errors = []

for tarea in tareas:
    audio_path = os.path.join(base, tarea)
    if os.path.isdir(audio_path):
        for folder in tqdm(os.listdir(audio_path)):
            for subfolder in os.listdir(os.path.join(audio_path, folder)):
                for file in os.listdir(os.path.join(audio_path, folder, subfolder)):
                    if not os.path.isdir(os.path.join(audio_path, folder, subfolder, file)):
                        try:
                            if preprocess:
                                processed_file = preprocess_audio(os.path.join(audio_path,
                                                                            folder, subfolder, file))
                            if diarize:
                                if processed_file.endswith(".wav"):
                                    if not os.path.exists(os.path.join(audio_path, folder, subfolder, "diarization_labels")):
                                        os.makedirs(os.path.join(audio_path, folder, subfolder, "diarization_labels"))
                                    wav_path = os.path.join(audio_path, processed_file)
                                    label_path = os.path.join(audio_path, folder, subfolder, "diarization_labels", file)
                                    audio_name = label_path[:-4]
                                    diarization = pipeline(wav_path)

                                    # print the result
                                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                                        # print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

                                        with open(f"{label_path}_labels.txt", "a", encoding="utf-8") as file:
                                            # 2. Use the write() method to append content
                                            label = f"{turn.start:.1f}	{turn.end:.1f}	{speaker}\n"
                                            file.write(label)
                        except Exception as e:
                            print("error, file not processed:", file)
                            print("Error message:", str(e))
                            errors.append(file, str(e))


    print(f"finished {tarea}")
    print("errors: ", errors)
