import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 定义4层全连接神经网络
class FCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.output_activation = nn.Sigmoid()  # 如果数据是0-1范围内，可以使用 Sigmoid 作为输出激活函数

    def forward(self, x):
        x = x.squeeze(0)
        # print("Init size: ", x.size())
        x = self.relu(self.fc1(x))
        # print("1 size: ", x.size())
        x = self.relu(self.fc2(x))
        # print("2 size: ", x.size())
        x = self.relu(self.fc3(x))
        # print("3 size: ", x.size())
        x = self.fc4(x)  # 输出层不再使用激活函数
        # print("Out size: ", x.size())
        return x


# 训练函数
def train(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()  # 清零梯度
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            running_loss += loss.item()

        # 打印每个 epoch 的训练损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    
    print("Training finished.")

# 测试函数

def plot_traffic_matrix(traffic_matrix, title="Traffic Matrix"):
    """绘制流量矩阵的热图。

    Args:
        traffic_matrix: 流量矩阵，形状为(num_nodes, num_nodes)
        title: 热图的标题
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(traffic_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Traffic Intensity')
    plt.title(title)
    plt.xlabel('Destination Nodes')
    plt.ylabel('Source Nodes')
    plt.grid(False)
    plt.show()

def test(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            plot_traffic_matrix(outputs)
            print("output size: ", outputs.size())
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
    return val_loss / len(val_loader)

# 保存模型
def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# 加载模型
def load_model(model, path='model.pth'):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")


# # 初始化模型
# model = FCNN(num_nodes * num_nodes, num_nodes * num_nodes)

# # 定义损失函数与优化器
# criterion = nn.MSELoss()  # 使用均方误差损失函数
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 训练
# train(model, train_loader, criterion, optimizer, num_epochs=50)

# # 在验证集上评估模型
# test(model, val_loader, criterion)

# # 保存模型
# save_model(model, 'traffic_model.pth')

# 如果需要加载模型
# load_model(model, 'traffic_model.pth')
