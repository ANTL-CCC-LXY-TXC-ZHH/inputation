import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num):
        """Initialize the FaultNetWork with the network structure.

        Args:
            input_dim: dimension of input data, history len * flattened traffic matrix
            output_dim: dimension of output data, len of candidate paths all s-d pairs
            layer_num: number of hidden layers
        """
        super(FCNN, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(layer_num):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.layers)
    
    def forward(self, x):
        """Forward the input data through the network.

        Args:
            x: input data, history len * flattened traffic matrix
        """
        x = x.squeeze(0)
        logits = self.net(x)
        return logits

# 4层GCN
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, 512)
        self.conv2 = GCNConv(512, 512)
        self.conv3 = GCNConv(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        # x = self.relu(self.conv3(x, edge_index))
        x = self.conv3(x, edge_index)

        return x


def plot_traffic_matrix(traffic_matrix, title="Traffic Matrix"):
    """绘制流量矩阵的热图。"""
    plt.figure(figsize=(10, 8))
    plt.imshow(traffic_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Traffic Intensity')
    plt.title(title)
    plt.xlabel('Destination Nodes')
    plt.ylabel('Source Nodes')
    plt.grid(False)
    plt.show()

# 训练函数
def train_FCN(model, train_loader, criterion, optimizer, scheduler, num_epochs=50, device=device):
    model.to(device)  # 将模型移到设备
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, targets = data  # 从DataLoader中获取输入和标签
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移到设备
            optimizer.zero_grad()  # 清零梯度
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            mae_value = mean_absolute_error(outputs, targets)
            # rmse_value = rmse(outputs, targets)
            loss = criterion(outputs, targets)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            running_loss += loss.item()
            
        scheduler.step(running_loss/len(train_loader))
        # 打印每个 epoch 的训练损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, MAE:{mae_value:.2f}')
    
    print("Training finished.")

# 测试函数
def test_FCN(model, val_loader, criterion, device=device):
    model.to(device)  # 将模型移到设备
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移到设备
            outputs = model(inputs)
            out_cpu = outputs.cpu()
            plot_traffic_matrix(out_cpu)
            mae_value = mean_absolute_error(outputs, targets)
            print("MAE: ", mae_value)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
    return val_loss / len(val_loader)

# 训练函数
def train_GCN(model, train_loader, criterion, optimizer,scheduler, edge_index, num_epochs, device=device):
    model.to(device)  # 将模型移到设备
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        count = 0
        for data in train_loader:
            inputs, targets = data  # 从DataLoader中获取输入和标签
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移到设备
            edge = edge_index[count].to(device)
            optimizer.zero_grad()  # 清零梯度
            # 前向传播
            outputs = model(inputs, edge)
            # 计算损失
            loss = criterion(outputs, targets)

            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # for name, parms in model.named_parameters():	
            #     print('-->name:', name)
            #     print('-->grad_requirs:',parms.requires_grad)
            #     print('-->grad_value:',parms.grad)
            #     print("===")
            running_loss += loss.item()
            count += 1
        scheduler.step(running_loss/len(train_loader))
        # 打印每个 epoch 的训练损失
        # if(epoch%10 == 0):
        #     output_cpu = outputs[0, : ,:].cpu()
        #     output = output_cpu.detach().numpy()
        #     plot_traffic_matrix(output)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    
    print("Training finished.")

# 测试函数
def test_GCN(model, val_loader, criterion, edge_index, device=device):
    model.to(device)  # 将模型移到设备
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        count = 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移到设备
            edge = edge_index[count].to(device)
            outputs = model(inputs, edge)       
            output_cpu = outputs[0, : ,:].cpu()
            target_cpu = targets[0, : ,:].cpu()
            MAE = mean_absolute_error(output_cpu, target_cpu)
            # output_cpu = output_cpu.squeeze(0)
            # plot_traffic_matrix(output_cpu)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            count += 1
    print(f'Validation Loss: {val_loss/len(val_loader):.4f}, MAE: {MAE:.4f}')
    return val_loss / len(val_loader)

# 保存模型
def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# 加载模型
def load_model(model, path='model.pth'):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
