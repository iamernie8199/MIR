# %%
from tqdm import tqdm
from glob import glob
import librosa
import librosa.display
import pandas as pd
import numpy as np
from madmom.audio.signal import Signal
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor

# AI
FILES = glob('AIVA/*/*.wav')
FILES += glob('Auxuman/*/*.wav')
FILES += glob('Auxuman/*/*.mp3')
FILES += glob('Yating/*/*.mp3')
# human
FILES += glob('else/*/*.wav')
FILES += glob('else/*/*.mp3')
FILES = [f.replace('\\', '/') for f in FILES]

vocal = ['ANIMA', 'French Kiwi Juice', 'In Rainbows']

df = pd.DataFrame(columns=[
    'track', 'album', 'length', 'key', 'tempo', 'loudness', 'dynamic_range',
    'volume', 'energy', 'zero_crossing_rate','type', 'by', 'artist'
])
df['zero_crossing_rate'] = df['zero_crossing_rate'].astype('object')
# %%
for f in tqdm(FILES):
    # f = FILES[0]
    y, sr = librosa.load(f)
    sig = Signal(f)
    frame = librosa.util.frame(y, frame_length=1024, hop_length=512)
    # length
    length = librosa.get_duration(y, sr)
    # zcr
    z = librosa.feature.zero_crossing_rate(y)[0]
    # key recognition
    proc = CNNKeyRecognitionProcessor()
    k = key_prediction_to_label(proc(f))
    # tempo
    proc = TempoEstimationProcessor(fps=100)
    tempi = proc(RNNBeatProcessor()(f))
    t = tempi[0][0] if tempi[0][1] - tempi[1][1] > 0.1 else max(
        tempi[0][0], tempi[1][0])
    # loudness
    S = librosa.stft(y)**2
    power = np.abs(S)**2
    p_mean = np.sum(power, axis=0, keepdims=True)
    p_ref = np.max(power)
    loudness = librosa.power_to_db(p_mean, ref=p_ref)
    # Volumns and energy
    v = np.mean(
        librosa.feature.rms(y, frame_length=1024, hop_length=512,
                            center=False))
    e = np.mean(np.sum(frame**2, axis=0))
    if f.split('/')[0] == 'Auxuman' or f.split('/')[1] in vocal:
        type0 = 'vocal'
    elif f.split('/')[1] == '艾娲 (Vol. 3 from artificial composer Aiva)':
        type0 = 'china'
    else:
        type0 = 'instrument'
    df = df.append([{
        'track': f.split('/')[-1].split('.wav')[0],
        'album': f.split('/')[1],
        'length': length,
        'key': k,
        'tempo': t,
        'loudness': loudness.mean(),
        'dynamic_range': loudness.max() - loudness.min(),
        'volume': v,
        'energy': e,
        'zero_crossing_rate': z,
        'type': type0,
        'by': 'human' if f.split('/')[0] == 'else' else 'AI',
        'artist': f.split('/')[0] if f.split('/')[0] != 'else' else 'else'
    }],ignore_index=True)

df.to_csv('data.csv', index=0)

# %%
