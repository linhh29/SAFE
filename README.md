This is the official pytorch implementation of the paper "Spatial-aware Semi-supervision for Arable Land Quality Evaluation".


1. Download the data and put it in the /data/

2. Revise the config file in ./SAFE/config.json

3. Data preprocess:
   - python preprocess.py
   - python get_spatialdata.py
   - python split_dataset.py 

4. Train and evaluate the model:
   - python ./main.py



