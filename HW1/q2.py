#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Example code for key detection preprocessing. You may start with filling the
'?'s below. There're also some description and hint within comment. However,
please feel free to modify anything as you like!

@author: selly
"""
from glob import glob
from collections import defaultdict
from librosa.feature import chroma_stft, chroma_cqt, chroma_cens
from scipy.stats import pearsonr
from mir_eval.key import weighted_score
from sklearn.metrics import accuracy_score
import numpy as np

# %%
import utils  # self-defined utils.py file

gamma = 1000
#DB = 'GTZAN'
DB = 'giantsteps'
if DB == 'GTZAN':  # dataset with genre label classify at parent directory
    FILES = glob(DB + '/wav/*/*.wav')
else:
    FILES = glob(DB + '/wav/*.wav')

GENRE = [g.replace('\\', '/').split('/')[2] for g in glob(DB + '/wav/*')]
n_fft = 100  # (ms)
hop_length = 25  # (ms)

# %% Q1
if DB == 'GTZAN':
    label, pred = defaultdict(list), defaultdict(list)
else:
    label, pred = list(), list()
chromagram = list()
gens = list()
for f in FILES:
    f = f.replace('\\', '/')
    if DB == 'GTZAN':
        content = utils.read_keyfile(f, '*.lerch.txt')
    else:
        content = int(utils.str_to_lerch(utils.generalize_key(utils.read_keyfile(f, '*.key'))))
    if (int(content) < 0): continue  # skip saving if key not found
    if DB == 'GTZAN':
        gen = f.split('/')[2]
        label[gen].append(utils.LABEL[int(content)])
        gens.append(gen)
    else:
        label.append(utils.LABEL[content])

    sr, y = utils.read_wav(f)
    ##########
    # TODO: Follow task1 description to give each audio file a key prediction.

    # compute the chromagram of audio data `y`
    cxx = chroma_stft(y=y, sr=sr)
    chromagram.append(cxx)  # store into list for further use
    # summing up all the chroma features into chroma vector
    chroma_vector = np.sum(cxx, axis=1)
    # Q2
    chroma_vector = np.log10(1 + gamma * chroma_vector)

    # finding the maximal value in the chroma vector and considering the note
    # name corresponding to the maximal value as the tonic pitch
    key_ind = int(np.argmax(chroma_vector))

    # finding the correlation coefficient between the summed chroma vectors and
    # the mode templates
    # Hint: utils.rotate(ar,n) may help you find different key mode template
    MODE = {"major": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            "minor": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]}
    KEY = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    mode = KEY[key_ind] + ' major' if pearsonr(chroma_vector, utils.rotate(MODE['major'], key_ind)) > pearsonr(
        chroma_vector, utils.rotate(MODE['minor'], key_ind)) else KEY[key_ind] + ' minor'

    if DB == 'GTZAN':
        pred[gen].append(mode)
    else:
        pred.append(mode)  # you may ignore this when starting with GTZAN dataset
##########

print("***** Q2 *****")
print(DB)
print("γ = "+str(gamma))
if DB == 'GTZAN':
    label_list, pred_list = list(), list()
    print("Genre    \taccuracy")
    for g in GENRE:
        ##################################################
        # TODO: Calculate the accuracy for each genre
        # Hint: Use label[g] and pred[g]
        ##################################################
        acc = accuracy_score(label[g], pred[g])
        print("{:9s}\t{:8.2f}%".format(g, acc * 100))
        label_list += label[g]
        pred_list += pred[g]
    print("----------")
else:
    label_list = label
    pred_list = pred

##################################################
# TODO: Calculate the accuracy for all file.
# Hint1: Use label_list and pred_list.
##################################################
acc_all = accuracy_score(label_list, pred_list)
print("Overall accuracy:\t{:.2f}%".format(acc_all * 100))
