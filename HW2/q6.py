from tqdm import tqdm
from numba import jit
from glob import glob
from collections import defaultdict
import numpy as np
from mir_eval.beat import f_measure
from madmom.features.beats import RNNBeatProcessor
from madmom.features.beats import BeatDetectionProcessor
from madmom.features.downbeats import RNNDownBeatProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor

db = 'Ballroom'
# db = 'JCS'
# db = 'SMC_MIREX'
if db == 'Ballroom':
    FILES = glob(db + '/wav/*/*.wav')
    GENRE = [g.replace('\\', '/').split('/')[2] for g in glob(db + '/wav/*')]
    F_b, F_db = defaultdict(list), defaultdict(list)
    gens = list()
elif db == 'SMC_MIREX':
    FILES = glob(db + '/SMC_MIREX_Annotations_05_08_2014/*.txt')
    F_b = list()
elif db == 'JCS':
    FILES = glob(db + '/audio/*.mp3')
    F_b, F_db = list(), list()
FILES = [f.replace('\\', '/') for f in FILES]

for f in tqdm(FILES):
    if db == 'Ballroom':
        beat = open(f.replace('/wav/', '/key_beat/').replace('.wav', '.beats'), 'r').read().split('\n')
        if beat[-1] == '':
            beat.pop()
        downbeat = [float(b.split()[0]) for b in beat if int(b.split()[1]) == 1]
        beat = [float(b.split('\t')[0]) for b in beat] if beat[0][-2] == '\t' else [float(b.split(' ')[0]) for b in
                                                                                    beat]
        gen = f.split('/')[2]
        gens.append(gen)
        downbeat = np.array(downbeat)
    elif db == 'SMC_MIREX':
        beat = open(f, 'r').read().split('\n')
        if beat[-1] == '':
            beat.pop()
        beat = [float(b) for b in beat]
    elif db == 'JCS':
        beat = open(f.replace('/audio/', '/annotations/').replace('.mp3', '_beats.txt'), 'r').read().split('\n')
        if beat[-1] == '':
            beat.pop()
        downbeat = [float(b.split()[0]) for b in beat if int(b.split()[1]) == 1]
        beat = [float(b.split('\t')[0]) for b in beat] if beat[0][-2] == '\t' else [float(b.split(' ')[0]) for b in
                                                                                    beat]
        downbeat = np.array(downbeat)
    beat = np.array(beat)

    proc1 = BeatDetectionProcessor(fps=100)
    if db == 'SMC_MIREX':
        file = f.replace('/SMC_MIREX_Annotations_05_08_2014/', '/SMC_MIREX_Audio/').replace('txt', 'wav')
        file = file[:-12] + file[-4:]
        act1 = RNNBeatProcessor()(file)
    else:
        act1 = RNNBeatProcessor()(f)
    beats = proc1(act1)
    if db != 'SMC_MIREX':
        proc2 = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        act2 = RNNDownBeatProcessor()(f)
        temp = proc2(act2)
        downbeats = [temp[i][0] for i in range(len(temp)) if (temp[i][1]) == 1]
        downbeats = np.array(downbeats)
        f_score_db = f_measure(downbeat, downbeats)
    f_score = f_measure(beat, beats)

    if db == 'Ballroom':
        F_b[gen].append(f_score)
        F_db[gen].append(f_score_db)
    elif db == 'JCS':
        F_b.append(f_score)
        F_db.append(f_score_db)
    else:
        F_b.append(f_score)

print("***** Q6 *****")
F_list, F_list_db = list(), list()
print('Dataset: %s' % (db))
if db == 'Ballroom':
    print('beat tracking')
    print("Genre    \tF-scores")
    for g in GENRE:
        acc = sum(F_b[g]) / len(F_b[g])
        print("{:9s}\t{:8.2f}".format(g, acc))
        F_list += F_b[g]
    print('downbeat tracking')
    for g in GENRE:
        acc = sum(F_db[g]) / len(F_db[g])
        print("{:9s}\t{:8.2f}".format(g, acc))
        F_list_db += F_db[g]
elif db == 'JCS':
    F_list = F_b
    F_list_db = F_db
else:
    F_list = F_b
acc_F_all = sum(F_list) / len(F_list)
print("----------")
print("Overall F-scores:\t{:.2f}".format(acc_F_all))
if db != 'SMC_MIREX':
    acc_F_all_db = sum(F_list_db) / len(F_list_db)
    print("Overall F-scores for downbeat tracking:\t{:.2f}".format(acc_F_all_db))
