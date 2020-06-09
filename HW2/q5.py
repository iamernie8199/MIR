from tqdm import tqdm
from numba import jit
import librosa
from glob import glob
from collections import defaultdict
import numpy as np
from mir_eval.beat import f_measure

# db = 'JCS'
db = 'SMC_MIREX'
if db == 'SMC_MIREX':
    FILES = glob(db + '/SMC_MIREX_Annotations_05_08_2014/*.txt')
FILES = [f.replace('\\', '/') for f in FILES]
F = list()

for f in tqdm(FILES):
    if db == 'SMC_MIREX':
        beat = open(f, 'r').read().split('\n')
    if beat[-1] == '':
        beat.pop()
    beat = [float(b) for b in beat]
    beat = np.array(beat)
    if db == 'SMC_MIREX':
        file = f.replace('/SMC_MIREX_Annotations_05_08_2014/','/SMC_MIREX_Audio/').replace('txt','wav')
        file = file[:-12] + file[-4:]
    y, sr = librosa.load(file)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beats = librosa.frames_to_time(beats, sr=sr)
    f_score = f_measure(beat, beats)
    F.append(f_score)

print("***** Q5 *****")
acc = sum(F) / len(F)
print("DB: [%s]" % (db))
print("Overall F-scores:\t{:.2f}".format(acc))
