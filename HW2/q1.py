from glob import glob
from collections import defaultdict
import librosa
import librosa.display
import numpy as np


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
label, pred = defaultdict(list), defaultdict(list)
P, ALOTC = defaultdict(list), defaultdict(list)
for f in FILES:
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
    label[gen].append(content)
    gens.append(gen)

    y, sr = librosa.load(f)

    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, y=y, sr=sr, hop_length=hop_length)
    """
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)[0]
    dtempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, aggregate=None)
    dtempo = [int(d) for d in dtempo]
    T1 = np.argmax(np.bincount(dtempo))
    tempi.append(T1)
    dtempo_ = np.bincount(dtempo)
    dtempo_[np.argmax(dtempo_)] = np.min(np.bincount(dtempo))
    T2 = np.argmax(dtempo_)
    tempi.append(T2)
    """
    bpms = librosa.core.tempo_frequencies(384, hop_length=hop_length, sr=sr)
    tg = [sum(t) for t in tempogram]
    for x in range(len(tg)):
        tg[x] = 0 if bpms[x] > 320 else tg[x]
    T1 = bpms[tg.index(max(tg))]
    tg.remove(max(tg))
    tempi.append(T1)
    T2 = bpms[tg.index(max(tg))]
    tempi.append(T2)
    idx = [list(bpms).index(x) for x in tempi]
    saliency = sum(tempogram[idx[0]]) / (sum(tempogram[idx[1]]) + sum(tempogram[idx[0]]))

    g = np.mean(60 / np.diff(beat))

    p_score = saliency * tt(T1, g) + (1-saliency) * tt(T2, g)
    ALOTC_score = 1 if tt(T1, g) == 1 or tt(T2, g) == 1 else 0

    pred[gen].append(tempi)
    P[gen].append(p_score)
    ALOTC[gen].append(ALOTC_score)

##########
print("***** Q1 *****")
label_list, pred_list = list(), list()
P_list, ALOTC_list = list(), list()
print("Genre    \tP-scores")
for g in GENRE:
    acc = sum(P[g]) / len(P[g])
    print("{:9s}\t{:8.2f}".format(g, acc))
    label_list += label[g]
    pred_list += pred[g]
print("\n")
print("Genre    \tALOTC-scores")
for g in GENRE:
    acc = sum(ALOTC[g]) / len(ALOTC[g])
    print("{:9s}\t{:8.2f}".format(g, acc))
    label_list += label[g]
    pred_list += pred[g]
    P_list += P[g]
    ALOTC_list += ALOTC[g]

acc_P_all = sum(P_list) / len(P_list)
acc_ALOTC_all = sum(ALOTC_list) / len(ALOTC_list)
##########
print("----------")
print("Overall P-scores:\t{:.2f}".format(acc_P_all))
print("Overall ALOTC-scores:\t{:.2f}".format(acc_ALOTC_all))
# librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo')
