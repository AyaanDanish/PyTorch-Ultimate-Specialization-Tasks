#%% package import
import torchaudio
from plot_audio  import plot_specgram
import os
import random
# %%
wav_path = 'data/set_a'
wav_filenames = os.listdir(wav_path)
random.shuffle(wav_filenames)
# %%
ALLOWED_CLASSES = ['normal', 'murmur', 'extrahls', 'artifact']
for file in wav_filenames:
    wav_class = file.split('__')[0]
    file_index = wav_filenames.index(file)

    target_path = 'train' if file_index < 140 else 'test'
    class_path = f"{target_path}/{wav_class}"
    file_path = f"{wav_path}/{file}"
    file_basename = os.path.basename(file)
    file_basename_wo_ext = os.path.splitext(file_basename)[0]
    target_file_path = f"{class_path}/{file_basename_wo_ext}.png"
    if (wav_class in ALLOWED_CLASSES):
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        data_waveform, sr = torchaudio.load(file_path)
        plot_specgram(waveform=data_waveform, sample_rate=sr, file_path=target_file_path)
        

# %%
