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
que = ['T/2', 'T/3', '2T', '3T']

hop_length = 512  # (ms)
gens = list()
P = defaultdict(lambda: defaultdict(list))
ALOTC = defaultdict(lambda: defaultdict(list))
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

    gens.append(gen)

    y, sr = librosa.load(f)

    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, y=y, sr=sr, hop_length=hop_length)
    g = np.mean(60 / np.diff(beat))
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
    T2 = bpms[tg.index(max(tg))]
    i, = np.where(bpms == T1)
    j, = np.where(bpms == T2)
    """
    type = "T1,T2"
    tempi.append(T1)
    tempi.append(T2)
    """
    T1_2 = bpms[i[0]*2]
    T2_2 = bpms[j[0]*2]
    tempi.append(T1_2)
    tempi.append(T2_2)
    idx = [list(bpms).index(x) for x in tempi]
    saliency = sum(tempogram[idx[0]]) / (sum(tempogram[idx[1]]) + sum(tempogram[idx[0]]))
    p_score = saliency * tt(T1_2, g) + (1 - saliency) * tt(T2_2, g)
    ALOTC_score = 1 if tt(T1_2, g) == 1 or tt(T2_2, g) == 1 else 0
    P['T/2'][gen].append(p_score)
    ALOTC['T/2'][gen].append(ALOTC_score)

    tempi.clear()
    T1_3 = bpms[i[0]*3]
    T2_3 = bpms[j[0]*3]
    tempi.append(T1_3)
    tempi.append(T2_3)
    idx = [list(bpms).index(x) for x in tempi]
    saliency = sum(tempogram[idx[0]]) / (sum(tempogram[idx[1]]) + sum(tempogram[idx[0]]))
    p_score = saliency * tt(T1_3, g) + (1 - saliency) * tt(T2_3, g)
    ALOTC_score = 1 if tt(T1_3, g) == 1 or tt(T2_3, g) == 1 else 0
    P['T/3'][gen].append(p_score)
    ALOTC['T/3'][gen].append(ALOTC_score)

    tempi.clear()
    T12 = bpms[int(round(i[0]/2))]
    T22 = bpms[int(round(j[0]/2))]
    tempi.append(T12)
    tempi.append(T22)
    idx = [list(bpms).index(x) for x in tempi]
    saliency = sum(tempogram[idx[0]]) / (sum(tempogram[idx[1]]) + sum(tempogram[idx[0]]))
    p_score = saliency * tt(T12, g) + (1 - saliency) * tt(T22, g)
    ALOTC_score = 1 if tt(T12, g) == 1 or tt(T22, g) == 1 else 0
    P['2T'][gen].append(p_score)
    ALOTC['2T'][gen].append(ALOTC_score)

    tempi.clear()
    T13 = bpms[int(round(i[0]/3))]
    T22 = bpms[int(round(j[0]/3))]
    tempi.append(T13)
    tempi.append(T22)
    idx = [list(bpms).index(x) for x in tempi]
    saliency = sum(tempogram[idx[0]]) / (sum(tempogram[idx[1]]) + sum(tempogram[idx[0]]))
    p_score = saliency * tt(T13, g) + (1 - saliency) * tt(T22, g)
    ALOTC_score = 1 if tt(T13, g) == 1 or tt(T22, g) == 1 else 0
    P['3T'][gen].append(p_score)
    ALOTC['3T'][gen].append(ALOTC_score)

##########
print("***** Q2 *****")
for q in que:
    print("Type: [%s]" % (q))
    P_list, ALOTC_list = list(), list()
    print("Genre    \tP-scores")
    for g in GENRE:
        acc = sum(P[q][g]) / len(P[q][g])
        print("{:9s}\t{:8.2f}".format(g, acc))
    print("\n")
    print("Genre    \tALOTC-scores")
    for g in GENRE:
        acc = sum(ALOTC[q][g]) / len(ALOTC[q][g])
        print("{:9s}\t{:8.2f}".format(g, acc))
        P_list += P[q][g]
        ALOTC_list += ALOTC[q][g]
    acc_P_all = sum(P_list) / len(P_list)
    acc_ALOTC_all = sum(ALOTC_list) / len(ALOTC_list)
    ##########
    print("----------")
    print("Overall P-scores:\t{:.2f}".format(acc_P_all))
    print("Overall ALOTC-scores:\t{:.2f}".format(acc_ALOTC_all))
    print("/////////////////////////")
