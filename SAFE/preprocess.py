import glob
import pandas as pd
import json
config = json.load(open('./config.json', 'r'))


root = config['data_dir'] + '/zx_kzsttj2019.csv'
files = glob.glob(root)
files.sort()
print(files)
sizes = []
for file in files:
    year = file.split('.')[0][-4:]
    print(year)
    data = pd.read_csv(file)
    
    geohashes = list(data['geohash'])
    with open(config['data_dir'] + f'/geohashes.txt', 'w') as f:
        f.write('\n'.join(geohashes))


    # 去除这三个变量： 'geohash', 'geohash_ejz', 'sj', 'kzstmj', 以及索引列 'Unnamed: 0'
    # 2019-2021
    continous_features = ['gdmj', 'ydmj', 'ldmj', 'cdmj', 'sdmj', 'nyssjsydmj', 'jtysydmj', 'czjsydmj', 'cunzjsydmj', 'sgssydmj', 'ldsy', 'wlyd', 'tchd', 
                    'gdggmj', 'gdxtzs', 'yn_2017', 'yphx', 'gc', 'pd', 'rksl']
    categorial_features = ['gdmjdj', 'ydmjdj', 'ldmjdj', 'cdmjdj', 'sdmjdj', 'nyssjsydmjdj', 'jtysydmjdj', 'czjsydmjdj', 'cunzjsydmjdj', 'sgssydmjdj',
                     'ldsydj', 'wlyddj', 'trzd', 'trlx', 'nyjgyqyds', 'jzzfwzjl']

    
    # # 2022
    # continous_features = ['gdmj', 'ydmj', 'ldmj', 'cdmj', 'sdmj', 'nyssjsydmj', 'jtysydmj', 'czjsydmj', 'cunzjsydmj', 'sgssydmj','ldsy', 'wlyd', 'gdbhmb_2022',
    #                         'yn_2022', 'stbhhx_2022', 'czkfbj_2022','yphx', 'gc', 'pd', 'rksl', 'tchd', 'gdggmj', 'gdxtzs']
    # categorial_features = ['gdmjdj', 'ydmjdj', 'ldmjdj', 'cdmjdj', 'sdmjdj', 'nyssjsydmjdj', 'jtysydmjdj', 'czjsydmjdj', 'cunzjsydmjdj', 'sgssydmjdj', 'ldsydj',
    #                         'wlyddj', 'trlx', 'nyjgyqyds', 'jzzfwzjl', 'trzd']
    
    # 填补缺失值
    value = {
        'tchd':0,
        'trzd':1000,
        'gdggmj':0,
        'gdxtzs':0,
        'rksl': 0
    }
    data = data.fillna(value=value)
    total = continous_features + categorial_features
    with open(config['data_dir'] + f'/feature_sizes_{year}.txt', 'w') as f:
        feature_sizes = []
        for i in range(len(total)):
            feature = total[i]
            feature_sizes.append(str(len(list(set(data[feature])))))

        f.write(','.join(feature_sizes))
