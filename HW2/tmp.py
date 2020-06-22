import librosa
import librosa.display
import matplotlib.pyplot as plt

f = 'JCS/audio/021.mp3'
y, sr = librosa.load(f)
hop_length = 512
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,hop_length=hop_length)
tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)[0]
bpms = librosa.core.tempo_frequencies(384, hop_length=hop_length, sr=sr)
librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo')
plt.show()