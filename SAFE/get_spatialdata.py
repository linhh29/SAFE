import pandas as pd
import glob
from geohash import decode_exactly
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import PatchCollection
import json
config = json.load(open('./config.json', 'r'))

files = glob.glob(config['data_dir'] + '/zx_kzsttj2019.csv')
files.sort()
print(files)

for file in files:
    data = pd.read_csv(file)
    print(data.columns)
    latitude_0 = []
    longtitude_0 = []
    latitude_1 = []
    longtitude_1 = []
    latitude_2 = []
    longtitude_2 = []

    for geo in data['geohash']:
        lat, lon, _, _ = decode_exactly(geo)
        # Area 0: 20.4,21.2; 109.6,110.5
        if 20.4 < lat < 21.2 and 109.6 < lon < 110.5:
            latitude_0.append(lat)
            longtitude_0.append(lon)
        # Area 1: 21.7,22.35; 110.6,111.4
        elif 21.7 < lat < 22.35 and 110.6 < lon < 111.4:
            latitude_1.append(lat)
            longtitude_1.append(lon)
        # Area 2: 22.2,22.8; 113.1,113.7
        elif 22.2 < lat < 22.8 and 113.1 < lon < 113.7:
            latitude_2.append(lat)
            longtitude_2.append(lon)
    
    latitude_0 = list(set(latitude_0))
    longtitude_0 = list(set(longtitude_0))
    latitude_1 = list(set(latitude_1))
    longtitude_1 = list(set(longtitude_1))
    latitude_2 = list(set(latitude_2))
    longtitude_2 = list(set(longtitude_2))

    latitude_0.sort()
    longtitude_0.sort()
    latitude_1.sort()
    longtitude_1.sort()
    latitude_2.sort()
    longtitude_2.sort()

    final_name = config['data_dir'] + '/geohash_to_index.txt'

    with open(final_name, 'w') as f:
        first_line = ','.join([str(len(latitude_0)),str(len(longtitude_0)),str(len(latitude_1)),str(len(longtitude_1)),str(len(latitude_2)),str(len(longtitude_2))])
        f.write(first_line + '\n')
        for i in range(len(data)):
            line = data.loc[i]
            geo = line['geohash']
            lat, lon, _, _ = decode_exactly(geo)
            final_data = [geo]
            # Area 0: 20.4,21.2; 109.6,110.5
            if 20.4 < lat < 21.2 and 109.6 < lon < 110.5:
                final_data.append(0)
                final_data.append(latitude_0.index(lat))
                final_data.append(longtitude_0.index(lon))
            # Area 1: 21.7,22.35; 110.6,111.4
            elif 21.7 < lat < 22.35 and 110.6 < lon < 111.4:
                final_data.append(1)
                final_data.append(latitude_1.index(lat))
                final_data.append(longtitude_1.index(lon))
            # Area 2: 22.2,22.8; 113.1,113.7
            elif 22.2 < lat < 22.8 and 113.1 < lon < 113.7:
                final_data.append(2)
                final_data.append(latitude_2.index(lat))
                final_data.append(longtitude_2.index(lon))

            final_data = [str(fd) for fd in final_data]
            f.write(','.join(final_data))
            f.write('\n')
