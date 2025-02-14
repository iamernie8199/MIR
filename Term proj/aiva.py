# %%
from tqdm import tqdm
from glob import glob
import librosa
import librosa.display
import pandas as pd
import numpy as np
import sklearn
from madmom.audio.signal import Signal
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor
import matplotlib.pyplot as plt
import plot


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


# AI
FILES = glob('AI/*/*/*.wav')
FILES += glob('AI/*/*/*.mp3')
# human
FILES += glob('human/*/*/*.wav')
FILES += glob('human/*/*/*.mp3')
FILES = [f.replace('\\', '/') for f in FILES]

df = pd.DataFrame(columns=[
    'track', 'album', 'length', 'key', 'tempo', 'loudness', 'dynamic_range', 'volume', 'energy',
    'zero_crossing_rate', 'zero_crossing_rate_var', 'spectral_centroid', 'spectral_centroid_var',
    'spectral_rolloff', 'spectral_rolloff_var', 'type', 'by', 'artist'
])
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
    z_v = np.var(z)
    z = normalize(z)
    # Spectral centroid
    sc = librosa.feature.spectral_centroid(y, sr=sr, n_fft=1024, hop_length=512, center=False)[0]
    sc_v = np.var(z)
    sc = normalize(sc)
    # spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=1024, hop_length=512, center=False, roll_percent=0.9)[
        0]
    rolloff_v = np.var(rolloff)
    rolloff = normalize(rolloff)
    # chroma
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    # mfcc
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    # key recognition
    proc = CNNKeyRecognitionProcessor()
    k = key_prediction_to_label(proc(f))
    # tempo
    proc = TempoEstimationProcessor(fps=100)
    tempi = proc(RNNBeatProcessor()(f))
    t = tempi[0][0] if tempi[0][1] - tempi[1][1] > 0.1 else max(tempi[0][0], tempi[1][0])
    # loudness
    S = librosa.stft(y) ** 2
    power = np.abs(S) ** 2
    p_mean = np.sum(power, axis=0, keepdims=True)
    p_ref = np.max(power)
    loudness = librosa.power_to_db(p_mean, ref=p_ref)
    # Volumns and energy
    v = np.mean(
        librosa.feature.rms(y, frame_length=1024, hop_length=512,
                            center=False))
    e = np.mean(np.sum(frame ** 2, axis=0))
    if f.split('/')[1] == 'Auxuman' or f.split('/')[1] == 'Jukebox' or ((f.split('/')[0] == 'human') and (
            f.split('/')[1] != 'Ryuichi Sakamoto') and (f.split('/')[1] != 'else')):
        type0 = 'vocal'
    elif f.split('/')[2] == '艾娲 (Vol. 3 from artificial composer Aiva)':
        type0 = 'china'
    else:
        type0 = 'instrument'
    # 將list轉成str儲存
    df = df.append([{
        'track': f.split('/')[-1].split('.wav')[0],
        'album': f.split('/')[2],
        'length': length,
        'key': k,
        'tempo': t,
        'loudness': loudness.mean(),
        'dynamic_range': loudness.max() - loudness.min(),
        'volume': v,
        'energy': e,
        'zero_crossing_rate': z,
        'zero_crossing_rate_var': z_v,
        'spectral_centroid': sc,
        'spectral_centroid_var': sc_v,
        'spectral_rolloff': rolloff,
        'spectral_rolloff_var': rolloff_v,
        'type': type0,
        'by': f.split('/')[0],
        'artist': f.split('/')[1]
    }], ignore_index=True)

data = df.copy()
"""
# %%
plot.count(df)

plot.general(df[df['type'] != 'vocal'], "length", "Length(ms)")
plot.general(df[df['type'] == 'vocal'], "length", "Length(ms).V")

plot.general(df[df['type'] != 'vocal'], "tempo", "Tempo(bpm)")
plot.general(df[df['type'] == 'vocal'], "tempo", "Tempo(bpm).V")

plot.general(df[df['type'] != 'vocal'], "loudness", "Loudness(db)")
plot.general(df[df['type'] == 'vocal'], "loudness", "Loudness(db).V")

plot.general(df[df['type'] != 'vocal'], "volume", "RMS")
plot.general(df[df['type'] == 'vocal'], "volume", "RMS.V")

plot.general(df[df['type'] != 'vocal'], "energy", "Energy")
plot.general(df[df['type'] != 'vocal'], "energy", "Energy.V")

plot.general(df[df['type'] != 'vocal'], "zero_crossing_rate_var", "Var of Zero Crossing Rate")
plot.general(df[df['type'] != 'vocal'], "zero_crossing_rate_var", "Var of Zero Crossing Rate.V")

plot.general(df[df['type'] != 'vocal'], "spectral_centroid_var", "Var of Spectral Centroid")
plot.general(df[df['type'] == 'vocal'], "spectral_centroid_var", "Var of Spectral Centroid.V")

plot.general(df[df['type'] != 'vocal'], "spectral_rolloff_var", "Var of Spectral Rolloff")
plot.general(df[df['type'] == 'vocal'], "spectral_rolloff_var", "Var of Spectral Rolloff.V")

plot.stack(df[(df['type'] != 'vocal')], 'key', 'by', 'Key')
plot.stack(df[(df['by'] == 'AI') & (df['type'] != 'vocal')], 'key', 'artist', 'Key.AI')
plot.stack(df[(df['by'] == 'human') & (df['type'] != 'vocal')], 'key', 'artist', 'Key.Human')
plot.stack(df[(df['type'] == 'vocal')], 'key', 'by', 'Key')
plot.stack(df[(df['by'] == 'AI') & (df['type'] == 'vocal')], 'key', 'artist', 'Key.AI.V')
plot.stack(df[(df['by'] == 'human') & (df['type'] == 'vocal')], 'key', 'artist', 'Key.Human.V')

features = ['length', 'tempo', 'loudness', 'dynamic_range', 'volume', 'energy',
            'zero_crossing_rate_var', 'spectral_centroid_var', 'spectral_rolloff_var']
music_features_df = df[df['by'] == 'AI'][features]
music_features_norm_df = sklearn.preprocessing.minmax_scale(music_features_df)
cov_matrix = music_features_df.cov()
plot.lower_diag_matrix_plot(cov_matrix, 'Covariance Matrix.AI')
corr_matrix = music_features_df.corr()
plot.lower_diag_matrix_plot(corr_matrix, 'Correlation Matrix.AI')
music_features_df = df[df['by'] == 'human'][features]
music_features_norm_df = sklearn.preprocessing.minmax_scale(music_features_df)
cov_matrix = music_features_df.cov()
plot.lower_diag_matrix_plot(cov_matrix, 'Covariance Matrix.Human')
corr_matrix = music_features_df.corr()
plot.lower_diag_matrix_plot(corr_matrix, 'Correlation Matrix.Human')
"""
