from glob import glob
import os

FILES = glob('AI/*/*/*.mid')
FILES = [f.replace('\\', '/') for f in FILES]

for f in FILES:
    os.remove(f)