import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import time
import sys
from config import config

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=config['SPOTIPY_CLIENT_ID'],
                                                                         client_secret=config['SPOTIPY_CLIENT_SECRET']))
sp.trace = False

if len(sys.argv) > 1:
    artist_name = ' '.join(sys.argv[1:])
else:
    artist_name = 'AIVA'

results = sp.search(q=artist_name, limit=50)
tids = []
for i, t in enumerate(results['tracks']['items']):
    print(' ', i, t['name'])
    tids.append(t['uri'])

features = sp.audio_features(tids)
for feature in features:
    print(json.dumps(feature, indent=4))
    print()
    analysis = sp._get(feature['analysis_url'])
    print(json.dumps(analysis, indent=4))
    print()