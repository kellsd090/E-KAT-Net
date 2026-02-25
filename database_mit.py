import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

def prepare_ecg_data(file_path, batch_size=64):
    df = pd.read_csv(file_path, header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64) # 交叉熵要求标签为 Long 类型索引

    # 归一化到 [-1, 1] 适配样条网格范围
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y) # 形状 [Batch]

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

if __name__ == "__main__":
    loader = prepare_ecg_data('E:/database/archive/mitbih_train.csv')  # loader: batch 1369, torch.Size([64, 187])
    X_batch, y_batch = next(iter(loader)) # [64, 187] [64, 1]

    sample_data = X_batch[0]
    sample_label = y_batch[0]
    print(f"第一个样本的数据（187维）:\n{sample_data}")
    print(f"第一个样本的种类标签: {sample_label}")
