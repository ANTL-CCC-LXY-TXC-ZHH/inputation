# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import hashlib

# 定义读取文件夹路径
input_folder = "1000_times_hash_data/1000_times_hash_data/"  # Excel文件夹路径

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
        vector.extend(hash_values)
    return np.array(vector)

# 使用线性迭代给赋予历史信息
history_factor = 0.3  # 定义历史因子的权重
history_vector = None

# 获取文件夹中的所有Excel文件
files = sorted([f for f in os.listdir(input_folder) if f.endswith('.xlsx')])

encoded_outputs = []

for file in files:
    file_path = os.path.join(input_folder, file)
    df = pd.read_excel(file_path, header=None)

    # 流量输入输出矩阵在第二行和第三行，列数是66列
    input_matrix = df.iloc[1:3, :66]

    # 对矩阵进行哈希编码，并转换为向量
    current_vector = hash_matrix_to_vector(input_matrix)

    # 通过线性迭代赋予历史信息
    if history_vector is None:
        # 第一个时间片的编码
        encoded_vector = current_vector
    else:
        # 当前编码加入历史信息，并进行归一化处理
        encoded_vector = history_factor * history_vector + (1 - history_factor) * current_vector
        encoded_vector = encoded_vector / np.max(np.abs(encoded_vector))  # 归一化处理

    history_vector = encoded_vector  # 更新历史向量
    encoded_outputs.append(encoded_vector)

# 输出结果
for i, encoding in enumerate(encoded_outputs):
    print(f"Time Slice {i+1}: Encoded Vector: {encoding}")
