from glob import glob
from collections import defaultdict
import numpy as np
from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor
from tqdm import tqdm


def tt(T, G):
    if np.abs((G - T) / G) > 0.08:
        return 0
    else:
        return 1


db = 'Ballroom'
if db == 'Ballroom':
    FILES = glob(db + '/wav/*/*.wav')

FILES = [f.replace('\\', '/') for f in FILES]
GENRE = [g.replace('\\', '/').split('/')[2] for g in glob(db + '/wav/*')]

hop_length = 512  # (ms)
gens = list()
P, ALOTC = defaultdict(list), defaultdict(list)
for f in tqdm(FILES):
    tempi = list()
    content = open(f.replace('/wav/', '/key_tempo/').replace('.wav', '.bpm'), 'r').read().strip()
    beat = open(f.replace('/wav/', '/key_beat/').replace('.wav', '.beats'), 'r').read().split('\n')
    if beat[-1] == '':
        beat.pop()
    if beat[0][-2] == '\t':
        beat = [float(b.split('\t')[0]) for b in beat]
    else:
        beat = [float(b.split(' ')[0]) for b in beat]
    gen = f.split('/')[2]
    gens.append(gen)
    g = np.mean(60 / np.diff(beat))

    proc = TempoEstimationProcessor(fps=100)
    act = RNNBeatProcessor()(f)
    tempi = proc(act)

    T1 = tempi[0][0]
    T2 = tempi[1][0]
    saliency = (tempi[0][1] - tempi[-1][1]) / (tempi[0][1] + tempi[-1][1])
    p_score = saliency * tt(T1, g) + (1 - saliency) * tt(T2, g)
    ALOTC_score = 1 if tt(T1, g) == 1 or tt(T2, g) == 1 else 0

    P[gen].append(p_score)
    ALOTC[gen].append(ALOTC_score)

print("***** Q3 *****")
P_list, ALOTC_list = list(), list()
print("Genre    \tP-scores")
for g in GENRE:
    acc = sum(P[g]) / len(P[g])
    print("{:9s}\t{:8.2f}".format(g, acc))
print("\n")
print("Genre    \tALOTC-scores")
for g in GENRE:
    acc = sum(ALOTC[g]) / len(ALOTC[g])
    print("{:9s}\t{:8.2f}".format(g, acc))
    P_list += P[g]
    ALOTC_list += ALOTC[g]

acc_P_all = sum(P_list) / len(P_list)
acc_ALOTC_all = sum(ALOTC_list) / len(ALOTC_list)
##########
print("----------")
print("Overall P-scores:\t{:.2f}".format(acc_P_all))
print("Overall ALOTC-scores:\t{:.2f}".format(acc_ALOTC_all))
