import os
import numpy as np
from os.path import isfile, isdir, join

# Librosa for audio
import librosa

dataset_path = "/home/lblier/data/ESC-50/"
orig_sampling = 44100

data_files = {}
for cat in os.listdir(dataset_path) :
    if isdir(join(dataset_path, cat)):
        data_files[cat] = os.listdir(join(dataset_path, cat))
n = sum(len(r) for r in data_files.values())


cat, files = next(iter(data_files.items()))
data = {}
for cat, files in data_files.items():
    print("Loading category : "+cat)
    data[cat] = [librosa.load(join(dataset_path,cat, f),
                              sr=orig_sampling) \
                 for f in files]


data_aligned = {}
#sampling = 11000
sampling = 44100
length_audio = 5
m = sampling*length_audio

def align_signal(y,m):
    if len(y) > m:
        return y[:m]
    if len(y) < m:
        z = np.zeros(m, dtype=np.float32)
        q = m//len(y)
        for w in range(q):
            z[w*len(y):(w+1)*len(y)] = y
        z[q*len(y):] = y[:m-len(y)]    
        return z

for cat, l in data.items():
    data_aligned[cat] = [ \
                          (align_signal(librosa.core.resample(y,sr,sampling),
                                        m),
                           sampling) \
                          for (y,sr) in l if type(y) == np.ndarray \
    ]
    data_aligned[cat] = list(filter(lambda x: type(x[0])==np.ndarray,
                                    data_aligned[cat]))
    


i = 0
u = 0
y = np.zeros(n)
X = np.zeros((n, m),dtype=np.float32)
for cat, l in data_aligned.items():
    y[u:u+len(l)] = i
    X[u:u+len(l),:] = np.stack([x[0] for x in l], axis=0)
    i += 1
    u += len(l)


np.save("/home/lblier/preprocessed/input.npy", X)
np.save("/home/lblier/preprocessed/output.npy", y)
