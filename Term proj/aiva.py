from tqdm import tqdm
from glob import glob
import librosa
import pandas as pd
import numpy as np
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor

FILES = glob('AIVA/*/*.wav')
FILES = [f.replace('\\', '/') for f in FILES]
ALBUM = [g.replace('\\', '/').split('/')[1] for g in glob('AIVA/*')]
df = pd.DataFrame(columns=['track', 'album', 'length', 'key', 'tempo'])

for f in tqdm(FILES):
    y, sr = librosa.load(f)
    # length
    l = librosa.core.get_duration(y=y, sr=sr)
    z = librosa.feature.zero_crossing_rate(y)
    # key recognition
    proc = CNNKeyRecognitionProcessor()
    k = key_prediction_to_label(proc(f))
    # tempo
    proc = TempoEstimationProcessor(fps=100)
    tempi = proc(RNNBeatProcessor()(f))
    t = tempi[0][0]
    df = df.append([{
        'track': f.split('/')[-1].split('.wav')[0],
        'album': f.split('/')[1],
        'length': l,
        'key': k,
        'tempo': t
    }], ignore_index=True)
