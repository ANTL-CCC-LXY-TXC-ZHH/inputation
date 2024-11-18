# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import hashlib

# �����ȡ�ļ���·��
input_folder = "1000_times_hash_data/1000_times_hash_data/"  # Excel�ļ���·��

# ��ȡExcel�ļ������й�ϣ����
def hash_matrix_to_vector(matrix):
    vector = []
    for row in matrix.iterrows():
        row_str = row[1].to_string(header=False, index=False)
        hash_object = hashlib.sha256(row_str.encode())
        # ����ϣֵת��Ϊ�����С���������������������Ȳ���СԪ����ֵ
        hash_hex = hash_object.hexdigest()
        # ���ɵĹ�ϣ����Ϊ16���Ƶ�64���ַ�������ֱ���ȡÿ�˸��ַ���ת��Ϊ10������������ϣ����
        hash_values = [int(hash_hex[i:i+8], 16) for i in range(0, len(hash_hex), 8)] 
        vector.extend(hash_values)
    return np.array(vector)

# ʹ�����Ե�����������ʷ��Ϣ
history_factor = 0.3  # ������ʷ���ӵ�Ȩ��
history_vector = None

# ��ȡ�ļ����е�����Excel�ļ�
files = sorted([f for f in os.listdir(input_folder) if f.endswith('.xlsx')])

encoded_outputs = []

for file in files:
    file_path = os.path.join(input_folder, file)
    df = pd.read_excel(file_path, header=None)

    # ����������������ڵڶ��к͵����У�������66��
    input_matrix = df.iloc[1:3, :66]

    # �Ծ�����й�ϣ���룬��ת��Ϊ����
    current_vector = hash_matrix_to_vector(input_matrix)

    # ͨ�����Ե���������ʷ��Ϣ
    if history_vector is None:
        # ��һ��ʱ��Ƭ�ı���
        encoded_vector = current_vector
    else:
        # ��ǰ���������ʷ��Ϣ�������й�һ������
        encoded_vector = history_factor * history_vector + (1 - history_factor) * current_vector
        encoded_vector = encoded_vector / np.max(np.abs(encoded_vector))  # ��һ������

    history_vector = encoded_vector  # ������ʷ����
    encoded_outputs.append(encoded_vector)

# ������
for i, encoding in enumerate(encoded_outputs):
    print(f"Time Slice {i+1}: Encoded Vector: {encoding}")
