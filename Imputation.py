from encoder import load_matrices_from_excel, finite_field_encoder_with_lagrange_projection
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import FCNN, GCN, train_GCN, test_GCN, train_FCN, test_FCN, save_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from helper import getModelSize
# 设置随机种子以确保结果可复现
np.random.seed(3407)
torch.manual_seed(3407)

# 检查是否可以使用 CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#folder_path = 'All_origin_data\\All_origin_data'

folder_path = 'All_origin_data\\TELESAT'
T = 1000  # 时间步长
num_nodes = 298  # 节点数
origin_data_tensor = load_matrices_from_excel(folder_path, T, num_nodes)  # 流量矩阵

# 拉格朗日压缩
compressed_vectors = finite_field_encoder_with_lagrange_projection(origin_data_tensor, alpha=0.3, beta=0.7, p=101, compression_dim=24)  # 每个卫星上进行压缩后，合并的特征矩阵

compression_dim = 24 * 2  # 压缩后的维度
sample_size = 6  # 每个时间点随机采样的卫星数量（这里可以调整）

# 假设 compressed_vectors 是原始数据
# compressed_vectors = np.random.rand(T, num_nodes, compression_dim)

# 生成掩码
mask = np.zeros((T, num_nodes, compression_dim))  # 初始化掩码，全部为 0

# 按组轮询采样
groups = [list(range(i, min(i + sample_size, num_nodes))) for i in range(0, num_nodes, sample_size)]
num_groups = len(groups)  # 总组数

for t in range(T):
    # 当前时间点选择的组
    current_group = groups[t % num_groups]  # 根据时间点 t 的轮询选择组
    
    # 设置掩码
    for node in range(num_nodes):
        if node in current_group:
            mask[t, node, :] = 1  # 当前组的卫星特征保留
        else:
            mask[t, node, :] = 1e-8 # 其他卫星特征置为 0

for t in range(T):
    for i in range(num_nodes):
        for j in range(compression_dim):
            if(origin_data_tensor[t,i,j] == 0):
                origin_data_tensor[t,i,j] = 1e-8
# 对应的 features_with_mask 数据
features_with_mask = compressed_vectors * mask  # 只保留被采样的卫星的数据，其余为 0

def create_edge_index_from_traffic(traffic_matrix):
    edge_index = torch.nonzero(traffic_matrix > 1e-6, as_tuple=False).t()  # 查找所有非零流量的索引
    return edge_index

# 计算动态边索引
edge_indices = []
for t in range(T):
    traffic_matrix = origin_data_tensor[t]  # 获取每个时间步的流量矩阵
    edge_index = create_edge_index_from_traffic(torch.tensor(traffic_matrix, dtype=torch.float32))
    edge_indices.append(edge_index)

# 输出查看部分掩码
print(mask.shape)  # 形状应为 (T, num_nodes, compression_dim)
print(mask[0, :, :])  # 打印第一个时间点的掩码情况

INPUT = torch.tensor(features_with_mask, dtype=torch.float32)  # 采样的，合并的特征矩阵 形状：T*num_nodes*compression_dim
TARGET = torch.tensor(origin_data_tensor, dtype=torch.float32)  # 真实流量矩阵

# 自定义归一化函数：对输入和目标进行归一化处理
def custom_normalization(x, xmax):
    return torch.tensor((np.log(x + 1) / np.log(xmax.item())), dtype=torch.float32)

INPUT_max = torch.max(INPUT)
INPUT_normalized = custom_normalization(INPUT, INPUT_max)
TARGET_max = torch.max(TARGET)
TARGET_normalized = custom_normalization(TARGET, TARGET_max)

# 划分训练集与验证集
X_train, X_val, y_train, y_val = train_test_split(INPUT_normalized, TARGET_normalized, train_size=0.9, test_size=0.1, random_state=3407)

# 使用 DataLoader 将数据分批
batch_size = 2
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = False)

val_data = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle = False)

# 模式选择：1表示训练模式，2表示测试模式
mode = 1  # 设置为 1 进行训练，设置为 2 进行测试

# 初始化模型并将其转移到 GPU/CPU
# model = FCNN(compression_dim, 512, num_nodes, 4).to(device)
# model = GCN(compression_dim, num_nodes)
model = GCN(compression_dim, num_nodes)
# 定义损失函数与优化器
criterion = nn.L1Loss()  # 使用均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4 )
# optimizer = optim.SparseAdam(model.parameters(), lr=1e-3,betas=(0.9,0.999),eps=1e-8)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience = 30, verbose=True, cooldown=10)
# 训练或测试模式选择

if mode == 1:
    # 训练模式
    print("进入训练模式...")
    getModelSize(model)
    #train_FCN(model, train_loader, criterion, optimizer, scheduler, num_epochs=100, device=device)
    train_GCN(model, train_loader, criterion, optimizer, scheduler, edge_indices, num_epochs=500, device=device)
    # 保存训练好的模型
    save_model(model, 'traffic_model.pth')

elif mode == 2:
    # 测试模式
    print("进入测试模式...")
    # 加载已经训练好的模型
    model.load_state_dict(torch.load('traffic_model.pth'))
    getModelSize(model)
    # 在验证集上评估模型
    # test_FCN(model, val_loader, criterion, device=device)
    test_GCN(model, val_loader, criterion, edge_indices, device=device)

else:
    print("无效的模式选择，请选择 1 (训练模式) 或 2 (测试模式)！")
