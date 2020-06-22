from tqdm import tqdm
from numba import jit
from glob import glob
from collections import defaultdict
import numpy as np
import librosa
from mir_eval.beat import f_measure
from madmom.features.downbeats import RNNDownBeatProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor

db = 'JCS'

FILES = glob(db + '/audio/*.mp3')
FILES = [f.replace('\\', '/') for f in FILES]
F_b, F_db = list(), list()

for f in tqdm(FILES):
    beat = open(f.replace('/audio/', '/annotations/').replace('.mp3', '_beats.txt'), 'r').read().split('\n')
    if beat[-1] == '':
        beat.pop()
    downbeat = [float(b.split()[0]) for b in beat if int(b.split()[1]) == 1]
    beat = [float(b.split('\t')[0]) for b in beat] if beat[0][-2] == '\t' else [float(b.split(' ')[0]) for b in beat]
    downbeat = np.array(downbeat)
    beat = np.array(beat)
    y, sr = librosa.load(f)
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    bpms = librosa.core.tempo_frequencies(384, hop_length=hop_length, sr=sr)
