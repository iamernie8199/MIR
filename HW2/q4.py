from tqdm import tqdm
from numba import jit
import librosa
from glob import glob
from collections import defaultdict
import numpy as np
from mir_eval.beat import f_measure

db = 'Ballroom'
FILES = glob(db + '/wav/*/*.wav')
FILES = [f.replace('\\', '/') for f in FILES]
GENRE = [g.replace('\\', '/').split('/')[2] for g in glob(db + '/wav/*')]
F = defaultdict(list)
gens = list()

for f in tqdm(FILES):
    content = open(f.replace('/wav/', '/key_tempo/').replace('.wav', '.bpm'), 'r').read().strip()
    beat = open(f.replace('/wav/', '/key_beat/').replace('.wav', '.beats'), 'r').read().split('\n')
    if beat[-1] == '':
        beat.pop()
    beat = [float(b.split('\t')[0]) for b in beat] if beat[0][-2] == '\t' else [float(b.split(' ')[0]) for b in beat]
    beat = np.array(beat)
    gen = f.split('/')[2]
    gens.append(gen)

    y, sr = librosa.load(f)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beats = librosa.frames_to_time(beats, sr=sr)
    f_score = f_measure(beat, beats)
    F[gen].append(f_score)

print("***** Q4 *****")
F_list = list()
print("Genre    \tF-scores")
for g in GENRE:
    acc = sum(F[g]) / len(F[g])
    print("{:9s}\t{:8.2f}".format(g, acc))
    F_list += F[g]
acc_F_all = sum(F_list) / len(F_list)
print("----------")
print("Overall F-scores:\t{:.2f}".format(acc_F_all))