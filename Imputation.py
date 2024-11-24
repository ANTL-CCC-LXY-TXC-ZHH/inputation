from encoder import load_matrices_from_excel, finite_field_encoder_with_lagrange_projection
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import FCNN, train_GCN, test_GCN, train_FCN, test_FCN, save_model

# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

folder_path = 'All_origin_data\\All_origin_data'
T = 1000  # 时间步长
num_nodes = 66  # 节点数
origin_data_tensor = load_matrices_from_excel(folder_path, T) # 流量矩阵

# 拉格朗日压缩
compressed_vectors = finite_field_encoder_with_lagrange_projection(origin_data_tensor, alpha=0.3, beta=0.7, p=101, compression_dim=10) #每个卫星上进行压缩后，合并的特征矩阵

compression_dim = 10 * 2  # 压缩后的维度
mask = np.random.rand(T, num_nodes, compression_dim) < 0.8  
features_with_mask = compressed_vectors * ~mask  # 掩盖的部分被设置为 0

INPUT = torch.tensor(features_with_mask, dtype=torch.float32)  # 采样的，合并的。特征矩阵 形状：T*num_nodes*compression_dim
TARGET = torch.tensor(origin_data_tensor, dtype=torch.float32)  # 真实流量矩阵

# 自定义归一化函数：对输入和目标进行归一化处理
def custom_normalization(x, xmax):
    return torch.tensor((np.log(x + 1) / np.log(xmax.item())), dtype=torch.float32)

INPUT_max = torch.max(INPUT)
INPUT_normalized = custom_normalization(INPUT, INPUT_max)
TARGET_max = torch.max(TARGET)
TARGET_normalized = custom_normalization(TARGET, TARGET_max)

# 划分训练集与验证集
X_train, X_val, y_train, y_val = train_test_split(INPUT_normalized, TARGET_normalized, train_size=0.8, test_size=0.2, random_state=42)

# 使用 DataLoader 将数据分批
batch_size = 1
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

val_data = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# 模式选择：0表示测试模式，1表示训练模式
mode = 2  # 设置为 1 进行训练，设置为 2 进行测试

# 初始化模型
model = FCNN(compression_dim, num_nodes)

# 使用全连接图构建节点间关系
edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()  # 生成完全连接图的边索引
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # 对称连接

# 定义损失函数与优化器
criterion = nn.MSELoss()  # 使用均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练或测试模式选择
if mode == 1:
    # 训练模式
    print("进入训练模式...")
    train_FCN(model, train_loader, criterion, optimizer, num_epochs=50)

    # 保存训练好的模型
    save_model(model, 'traffic_model.pth')

elif mode == 2:
    # 测试模式
    print("进入测试模式...")
    # 加载已经训练好的模型
    model.load_state_dict(torch.load('traffic_model.pth'))

    # 在验证集上评估模型
    test_FCN(model, val_loader, criterion)

else:
    print("无效的模式选择，请选择 1 (测试模式) 或 2 (训练模式)！")
