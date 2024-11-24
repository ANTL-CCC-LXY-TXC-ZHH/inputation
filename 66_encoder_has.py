# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import hashlib

# 定义读取输出文件夹路径
input_folder= "All_origin_data/All_origin_data/"  # Excel文件夹路径
output_folder = "encoded_outputs/"  # 输出Excel文件夹路径

# 创建输出文件夹（encoded_outputs）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取Excel文件，进行哈希编码
def hash_matrix_to_vector(matrix):
    vector = []
    for row in matrix.iterrows():
        row_str = row[1].to_string(header=False, index=False)
        hash_object = hashlib.sha256(row_str.encode())
        # 将哈希值转换为多个较小的整数，以增加向量长度并减小元素数值
        hash_hex = hash_object.hexdigest()
        # 生成的哈希编码为16进制的64个字符，这里分别提取每八个字符并转化为10进制来构建哈希向量
        hash_values = [int(hash_hex[i:i+8], 16) for i in range(0, len(hash_hex), 8)] 
        vector.extend(hash_values[:16])  # 取前16个值作为编码向量
    return np.array(vector)

# 使用线性迭代给赋予历史信息
history_factor = 0.3  # 定义历史因子的权重
history_vectors = [None] * 66  # 初始化66个节点的历史向量

# 读取文件夹中的所有Excel文件
files = sorted([f for f in os.listdir(input_folder) if f.endswith('.xlsx')])

# 处理每个时间片的文件
for file_index, file in enumerate(files):
    file_path = os.path.join(input_folder, file)
    df = pd.read_excel(file_path, header=None)
    
    # 初始化每个时间片的编码结果
    encoded_results = []

    # 对每个节点进行编码，第i个节点的输入输出信息在第 i+1 行和 i+2 行，共有 66 个节点
    for node_index in range(66):
        input_matrix = df.iloc[node_index * 2 + 1 : node_index * 2 + 3, :66]

        # 对矩阵进行哈希编码，并转换为向量
        current_vector = hash_matrix_to_vector(input_matrix)

        # 通过线性迭代赋予历史信息
        if history_vectors[node_index] is None:
            # 第一个时间片的编码(无历史信息)
            encoded_vector = current_vector / np.max(np.abs(current_vector))  # 归一化处理
        else:
            # 当前编码加入历史信息，并归一化处理
            encoded_vector = history_factor * history_vectors[node_index] + (1 - history_factor) * current_vector
            encoded_vector = encoded_vector / np.max(np.abs(encoded_vector))  # 归一化处理

        # 更新历史向量
        history_vectors[node_index] = encoded_vector
        encoded_results.append(encoded_vector)

    # 将当前时间片的所有节点的编码结果保存到Excel
    output_data = pd.DataFrame(encoded_results)
    output_file_path = os.path.join(output_folder, f"time_slice_{file_index + 1}.xlsx")
    output_data.to_excel(output_file_path, header=False, index=False)

    print(f"Time Slice {file_index + 1}: Encoded data saved to {output_file_path}")

print("All slices have been saved.")

# 如何编码使得信息量不衰减——————>数学证明
# RNN
# 数学上的，时间/空间复杂度低的方案
# 信息论，群论