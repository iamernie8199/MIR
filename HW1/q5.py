#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example code for local key detection preprocessing. You may start with filling
the '?'s below. There're also some description and hint within comment. However,
please feel free to modify anything as you like!

@author: selly
"""
import numpy as np
from glob import glob
from librosa.feature import chroma_stft
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from mir_eval.key import weighted_score
import pretty_midi as pm

import utils  # self-defined utils.py file

#DB = 'BPS-FH'
DB = 'A-MAPS'

chromagram, label, index = list(), list(), list()
pred = list()
if DB == 'BPS-FH':
    for f in glob(DB + '/wav/*.wav'):
        f = f.replace('\\', '/')
        key = utils.parse_key([line.split('\t')[1] for line in utils.read_keyfile(f, '*.txt').split('\n')])
        label.extend(key)
        ind = int(f.split('/')[-1].strip('.wav'))
        sr, y = utils.read_wav(f)
        y_second = [list(i) for i in zip(*[iter(y)] * 44100)]
        for Q in range(len(y_second)):
            Y = np.array(y_second[Q])
            cxx = chroma_stft(y=Y, sr=sr)
            chromagram.append(cxx)
            chroma_vector = np.sum(cxx, axis=1)
            """
            # bin
            key_ind = int(np.argmax(chroma_vector))
            R_major = pearsonr(chroma_vector, utils.rotate(utils.MODE['major'], key_ind))[0]
            R_minor = pearsonr(chroma_vector, utils.rotate(utils.MODE['minor'], key_ind))[0]
            mode = utils.lerch_to_str((key_ind + 3) % 12 if R_major > R_minor else (key_ind + 3) % 12 + 12)
            """
            # K-S
            tmp = list()
            for i in range(12):
                x = pearsonr(chroma_vector, utils.rotate(utils.KS['major'], i))[0]
                tmp.append(x)
            for i in range(12):
                x = pearsonr(chroma_vector, utils.rotate(utils.KS['minor'], i))[0]
                tmp.append(x)
            idx = tmp.index(max(tmp))
            mode = utils.lerch_to_str((idx + 3) % 12 if idx < 12 else (idx + 3) % 12 + 12)
            pred.append(mode)
        if len(label) > len(pred):
            for i in range(len(label) - len(pred)):
                label.pop()
        elif len(label) < len(pred):
            for i in range(len(pred) - len(label)):
                pred.pop()
else:
    for f in glob(DB + '/*.mid'):
        f = f.replace('\\', '/')
        midi_data = pm.PrettyMIDI(f)
        key = midi_data.key_signature_changes
        for i in range(len(key)):
            t = round(key[i+1].time) - round(key[i].time) if i+1 < len(key) else round(midi_data.get_end_time())-round(key[i].time)
            label.extend([utils.parse_key_number(key[i].key_number)]*int(t))
        cxx = midi_data.get_chroma(1)
        for i in range(len(cxx[0])):
            """
            # bin
            key_ind = int(np.argmax(cxx[:, i]))
            R_major = pearsonr(cxx[:, i], utils.rotate(utils.MODE['major'], key_ind))[0]
            R_minor = pearsonr(cxx[:, i], utils.rotate(utils.MODE['minor'], key_ind))[0]
            mode = utils.lerch_to_str((key_ind + 3) % 12 if R_major > R_minor else (key_ind + 3) % 12 + 12)
            """
            #ks
            tmp = list()
            for j in range(12):
                x = pearsonr(cxx[:, i], utils.rotate(utils.KS['major'], i))[0]
                tmp.append(x)
            for j in range(12):
                x = pearsonr(cxx[:, i], utils.rotate(utils.KS['minor'], i))[0]
                tmp.append(x)
            idx = tmp.index(max(tmp))
            mode = utils.lerch_to_str((idx + 3) % 12 if idx < 12 else (idx + 3) % 12 + 12)
            pred.append(mode)
        if len(label) > len(pred):
            for i in range(len(label) - len(pred)):
                label.pop()
        elif len(label) < len(pred):
            for i in range(len(pred) - len(label)):
                pred.pop()
acc_all = accuracy_score(label, pred)
"""
tmp = 0
for f, b in zip(label, pred):
    x = weighted_score(f, b)
    tmp += x
acc_all = tmp/len(label)
"""
print("----------")
print("accuracy:\t{:.2f}%".format(acc_all * 100))
