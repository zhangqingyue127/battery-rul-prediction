import os
import random
import numpy as np
import scipy.io
import torch
from datetime import datetime

def setup_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def convert_to_time(hmm):
    y, m, d, H, M, S = map(int, hmm)
    return datetime(year=y, month=m, day=d, hour=H, minute=M, second=S)

def loadMat(matfile):
    filename = os.path.basename(matfile).split('.')[0]
    data = scipy.io.loadmat(matfile)
    col = data[filename][0][0][0][0]
    out = []
    for i in range(col.shape[0]):
        d1, d2 = {}, {}
        if str(col[i][0][0]) != 'impedance':
            keys = list(col[i][3][0].dtype.fields.keys())
            for j in range(len(keys)):
                t = col[i][3][0][0][j][0]
                d2[keys[j]] = [val.item() for val in t]
        d1['type'] = str(col[i][0][0])
        d1['temp'] = int(col[i][1][0])
        d1['time'] = str(convert_to_time(col[i][2][0]))
        d1['data'] = d2
        out.append(d1)
    return out

def getBatteryCapacity(Battery):
    cycle, capacity = [], []
    for i, Bat in enumerate(Battery):
        if Bat['type'] == 'discharge' and 'Capacity' in Bat['data']:
            capacity.append(Bat['data']['Capacity'][0])
            cycle.append(i + 1 if 'Cycle' not in Bat['data'] else Bat['data']['Cycle'][0])
    return [cycle, capacity]

def load_battery_data(data_dir, battery_list):
    """Load and preprocess battery data from MAT files"""
    npy_path = os.path.join(data_dir, "NASA_Battery_Data.npy")
    battery_data = {}
    
    if os.path.exists(npy_path):
        print(f"Loading cached data from {npy_path}")
        battery_data = np.load(npy_path, allow_pickle=True).item()
        return battery_data
    
    print(f"Processing MAT files from {data_dir}")
    for name in battery_list:
        mat_path = os.path.join(data_dir, f"{name}.mat")
        if not os.path.exists(mat_path):
            print(f"Warning: {mat_path} not found, skipping {name}")
            continue
        mat_data = loadMat(mat_path)
        battery_data[name] = getBatteryCapacity(mat_data)
    
    np.save(npy_path, battery_data)
    print(f"Cached data saved to {npy_path}")
    return battery_data