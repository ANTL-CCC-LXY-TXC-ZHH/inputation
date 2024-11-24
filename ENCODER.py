import pandas as pd
import numpy as np
import os

def load_matrices_from_excel(folder_path, num_files):
    """
    从指定文件夹中加载多个Excel文件，提取每个文件中的66x66矩阵，最终拼接成一个三维张量。
    
    参数：
        folder_path (str): 包含Excel文件的文件夹路径
        num_files (int): 文件夹中Excel文件的数量
    
    返回：
        np.ndarray: 一个三维张量，形状为 (num_files, 66, 66)
    """
    matrices = []
    
    for i in range(1, num_files + 1):
        # 构建文件路径，假设文件名为file_1.xlsx, file_2.xlsx, ..., file_1000.xlsx
        file_path = os.path.join(folder_path, f'{i}.xlsx')
        
        # 读取Excel文件
        try:
            df = pd.read_excel(file_path, header=None)  # 假设文件没有列名和索引
            matrix = df.to_numpy()  # 转换为NumPy数组
            
            if matrix.shape == (66, 66):
                matrices.append(matrix)
                print("Read done: ",i)
            else:
                print(f"警告: 文件 {file_path} 不是66x66矩阵，跳过该文件。")
        except Exception as e:
            print(f"错误: 读取文件 {file_path} 时发生异常: {e}")
    
    # 将所有矩阵拼接成一个三维张量
    return np.array(matrices)

def lagrange_interpolation(x_values, y_values, p):
    """
    在有限域 GF(p) 上通过拉格朗日插值法生成插值多项式的系数。
    
    参数：
        x_values (list): 插值节点 x_i
        y_values (list): 对应的 y_i
        p (int): 有限域的质数 p
        
    返回：
        list: 对应的插值多项式的系数
    """
    n = len(x_values)
    result = [0] * n
    
    # 计算每个基函数的系数
    for i in range(n):
        # 计算第i个基多项式l_i(x)
        li_numer = 1
        li_denom = 1
        for j in range(n):
            if i != j:
                li_numer = (li_numer * (x_values[i] - x_values[j])) % p
                li_denom = (li_denom * (x_values[i] - x_values[j])) % p
        
        # 计算l_i(x)的系数
        li_inv = pow(int(li_denom), p - 2, p)  # 使用 int() 确保类型转换
        li_coeff = li_numer * li_inv % p
        
        # 更新最终结果
        for j in range(n):
            result[j] = (result[j] + y_values[i] * li_coeff) % p
    
    return result

def finite_field_encoder_with_lagrange_projection(data_tensor, alpha=0.3, beta=0.7, p=101, compression_dim=10):
    """
    使用拉格朗日插值法生成压缩矩阵，并将每个节点的流量向量压缩到低维空间。
    
    参数：
        data_tensor (np.ndarray): 形状为 (num_times, num_nodes, num_nodes) 的三维张量
        p (int): 有限域的质数 p，默认设置为 101
        compression_dim (int): 压缩后的向量维度，默认设置为 10
    
    返回：
        np.ndarray: 压缩后的向量，形状为 (num_times, num_nodes, compression_dim)
    """
    num_times, num_nodes, _ = data_tensor.shape
    compressed_vectors = []

    # 随机选择 compression_dim 个插值节点 x_i
    x_values = np.random.randint(0, num_nodes, compression_dim)

    # 为每个节点生成插值基函数
    y_values = np.random.randint(0, p, compression_dim)  # 随机生成 y_i 值
    coefficients = lagrange_interpolation(x_values, y_values, p)

    # 构造拉格朗日插值矩阵 A (compression_dim x num_nodes)
    A = np.zeros((compression_dim, num_nodes), dtype=int)

    # 填充矩阵 A
    for i in range(compression_dim):
        for j in range(num_nodes):
            A[i, j] = coefficients[i]  # 使用插值系数填充矩阵

    for t in range(num_times):
        matrix = data_tensor[t]
        compressed_vector_last = [0]*2*compression_dim
        compressed_vectors_at_time_t = []
        for i in range(num_nodes):
            # 节点 i 的输出流量（第 i 行）
            output_flow = matrix[i, :]
            
            # 节点 i 的输入流量（第 i 列）
            input_flow = matrix[:, i]
            
            # 使用有限域 GF(p) 进行压缩编码
            input_flow = input_flow % p
            output_flow = output_flow % p

            # 分别对输入和输出流量进行映射
            compressed_input = np.dot(A, input_flow) % p
            compressed_output = np.dot(A, output_flow) % p
            
            # 将压缩后的输入输出流量拼接
            compressed_vector = np.concatenate([compressed_input, compressed_output])
            
            for i in range(len(compressed_vector)):
                compressed_vector[i] = alpha * compressed_vector_last[i] + beta * compressed_vector[i]

            compressed_vector_last = compressed_vector
            compressed_vectors_at_time_t.append(compressed_vector)
        compressed_vectors.append(compressed_vectors_at_time_t)
    
    # 返回压缩后的所有时刻的向量，形状为 (num_times, num_nodes, compression_dim * 2)
    return np.array(compressed_vectors)

folder_path = 'All_origin_data\All_origin_data'  # 这里替换成你存放Excel文件的文件夹路径
num_files = 100
data_tensor = load_matrices_from_excel(folder_path, num_files)
compressed_vectors = finite_field_encoder_with_lagrange_projection(data_tensor)

# 打印压缩后向量的形状，验证
print("压缩后的向量形状:", compressed_vectors.shape)
# print(compressed_vectors[:,:])

