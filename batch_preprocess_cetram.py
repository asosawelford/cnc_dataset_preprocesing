import os
from tqdm import tqdm
from preprocess import preprocess_audio

base = "/home/udesa_ubuntu/fondecyt/CETRAM/Participantes"
diarize = True
GPU = False
preprocess = False

if diarize:
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token="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")

    if GPU:
        import torch
        pipeline.to(torch.device("cuda"))


errors = []

for participante in tqdm(os.listdir(base)):
    participante_path = os.path.join(base, participante)
    for folder in os.listdir(participante_path):
        for file in os.listdir(os.path.join(participante_path, folder)):
            audio_path = os.path.join(participante_path, folder, file)
            if not os.path.isdir(audio_path):
                try:
                    if preprocess:
                        processed_file = preprocess_audio(audio_path)
                    if diarize:
                        if not os.path.exists(os.path.join(participante_path, folder, "diarization_labels")):
                            os.makedirs(os.path.join(participante_path, folder, "diarization_labels"))
                        label_path = os.path.join(participante_path, folder, "diarization_labels", file)
                        audio_name = label_path[:-4]
                        diarization = pipeline(audio_path)

                        # print the result
                        for turn, _, speaker in diarization.itertracks(yield_label=True):
                            # print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

                            with open(f"{label_path}_labels.txt", "a", encoding="utf-8") as file:
                                # 2. Use the write() method to append content
                                label = f"{turn.start:.1f}	{turn.end:.1f}	{speaker}\n"
                                file.write(label)
                except:
                    errors.append(file)
    print(f"finished {participante}")
print("errors: ", len(errors))
with open("erorrs.txt","w", encoding="utf-8") as f:
    for error in errors:
        f.write(error + "\n")
