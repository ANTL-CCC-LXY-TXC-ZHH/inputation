from  ENCODER import load_matrices_from_excel, finite_field_encoder_with_lagrange_projection
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import FCNN, train, test, save_model


# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

folder_path = 'All_origin_data\All_origin_data'
T = 100 # 时间步长
num_nodes = 66  # 节点数
origin_data_tensor = load_matrices_from_excel(folder_path, T)
compressed_vectors = finite_field_encoder_with_lagrange_projection(origin_data_tensor, alpha=0.3, beta=0.7, p=101, compression_dim=10)
compression_dim = 10*2
mask = np.random.rand(T, num_nodes, compression_dim) < 0.8  # 20%的数据随机掩盖
features_with_mask = compressed_vectors * ~mask  # 掩盖的部分被设置为 0
# print("压缩后的向量形状:", features_with_mask.shape)

INPUT = torch.tensor(features_with_mask, dtype=torch.float32)  # 形状：T*num_nodes*compression_dim
TARGET = torch.tensor(origin_data_tensor, dtype=torch.float32)  # 真实数据

def custom_normalization(x, xmax):
    return torch.tensor((np.log(x + 1) / np.log(xmax.item())), dtype=torch.float32)

INPUT_max = torch.max(INPUT)
INPUT_normalized = custom_normalization(INPUT, INPUT_max)
TARGET_max = torch.max(TARGET)
TARGET_normalized = custom_normalization(TARGET, TARGET_max)
print(TARGET_normalized.shape)
# 划分训练集与验证集
X_train, X_val, y_train, y_val = train_test_split(INPUT_normalized, TARGET_normalized, train_size=0.8, test_size=0.2, random_state=42)

# 使用 DataLoader 将数据分批
batch_size = 1
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

val_data = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# 初始化模型
model = FCNN(compression_dim, num_nodes)

# 定义损失函数与优化器
criterion = nn.MSELoss()  # 使用均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
train(model, train_loader, criterion, optimizer, num_epochs=50)

# 在验证集上评估模型
test(model, val_loader, criterion)

# 保存模型
save_model(model, 'traffic_model.pth')
