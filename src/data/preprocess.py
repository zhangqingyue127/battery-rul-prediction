import numpy as np

def build_instances(sequence, window_size):
    """Build sliding window features for time series prediction"""
    x, y = [], []
    for i in range(len(sequence) - window_size):
        x.append(sequence[i:i + window_size])
        y.append(sequence[i + window_size])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

def get_train_test(data_dict, name, window_size=16, train_split_ratio=0.5):
    """Split data into train/test sets and build features"""
    cycle_seq, data_seq = data_dict[name][0], data_dict[name][1]
    split_point = int(len(data_seq) * train_split_ratio)
    
    # Ensure split point is valid
    if split_point <= window_size:
        split_point = window_size + 1
    
    train_data = data_seq[:split_point]
    test_data = data_seq[split_point:]
    
    # Build features
    train_x, train_y = build_instances(train_data, window_size)
    test_x, test_y = build_instances(data_seq, window_size)
    
    # Merge data from other batteries for training
    for k, v in data_dict.items():
        if k != name:
            x2, y2 = build_instances(v[1], window_size)
            train_x = np.r_[train_x, x2]
            train_y = np.r_[train_y, y2]
    
    return train_x, train_y, train_data, test_data, cycle_seq, data_seq, test_x, test_y