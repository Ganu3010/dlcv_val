import numpy as np
import pandas as pd
from renderer import *

def create_sample(N, WINDOW, save_path, data_path = 'data/uptown_funk.json', seed = 42):                # Function to generate random samples. 
    var = load_keypoints(data_path)
    var = dict(var)
    df = pd.DataFrame.from_dict(var)
    fdf = pd.DataFrame()
    for col in df.columns:
        fdf[col+"_x"] = df[col].apply(lambda x: x[0])
    for col in df.columns:
        fdf[col+"_y"] = df[col].apply(lambda x: x[1])

    for col in fdf.columns:
        fdf[col] = (fdf[col] - min(fdf[col]))/(max(fdf[col]) - min(fdf[col]))
    
    samples = []
    for i in range(N):
        start = np.random.randint(low = 0, high = len(df)-WINDOW)
        samples.append(fdf.iloc[start:start+WINDOW])

    images = np.array(samples).reshape(N, 2, WINDOW, len(df.columns)).astype(np.float32)

    np.save(save_path, images)



def get_meta_data(data = 'data/uptown_funk.json'):                                                      # Function to get scaling data of 2D matrices. 
    var = load_keypoints(data)
    var = dict(var)
    df = pd.DataFrame.from_dict(var)
    fdf = pd.DataFrame()
    for col in df.columns:
        fdf[col+"_x"] = df[col].apply(lambda x: x[0])
    for col in df.columns:
        fdf[col+"_y"] = df[col].apply(lambda x: x[1])

    data = {}
    for col in fdf.columns:
        data[col] = {'min': min(fdf[col]), 'diff': max(fdf[col]) - min(fdf[col])}
        fdf[col] = (fdf[col] - min(fdf[col]))/(max(fdf[col]) - min(fdf[col]))   
    return data, df.columns