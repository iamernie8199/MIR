import numpy as np
from glob import glob
import librosa
import random
import librosa.display
from numba import jit
import matplotlib.pyplot as plt

# FILES = glob('AI/AIVA/*/*.wav')
# FILES = [f.replace('\\', '/') for f in FILES]
FILES = glob('AI/MuseNet/*/*.mp3')
FILES = [f.replace('\\', '/') for f in FILES]

i = random.randint(0, len(FILES))
print(FILES[i])
y, sr = librosa.load(FILES[i])
hop_length = 1024
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
R_aff = librosa.segment.recurrence_matrix(mfcc, mode='affinity', metric='cosine')
librosa.display.specshow(R_aff, x_axis='time', y_axis='time', hop_length=hop_length, cmap='gray_r')
plt.title(FILES[i].split('/')[-1] + '\nAffinity recurrence')
plt.show()
