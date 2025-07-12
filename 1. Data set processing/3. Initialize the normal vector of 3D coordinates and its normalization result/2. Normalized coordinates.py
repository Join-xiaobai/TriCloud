import numpy as np
import pandas as pd

# 定义函数读取txt文件
def read_data(file_path):
    # 假设文件是逗号分隔的，跳过第一行（列名）
    data = pd.read_csv(file_path, delimiter=',', skiprows=1, header=None)
    return data

# 归一化函数：将数据缩放到[0, 1]范围
def normalize_data(data):
    # 对每一列进行归一化
    normalized_data = (data - data.min()) / (data.max() - data.min())
    return normalized_data

# 保存归一化后的数据到新的txt文件
def save_normalized_data(data, output_file):
    # 保存结果到新的txt文件，不保存列名
    data.to_csv(output_file, sep=',', header=False, index=False, float_format='%.6f')
    print(f"归一化完成并已保存到 {output_file}")

# 主函数：读取、归一化、保存
def normalize_txt_file(input_file, output_file):
    # 读取数据
    data = read_data(input_file)
    
    # 归一化数据
    normalized_data = normalize_data(data)
    
    # 保存归一化后的数据
    save_normalized_data(normalized_data, output_file)

# 输入输出文件路径
input_file = 'drug_gene_disease_coordinate.txt'  # 替换为你的输入文件路径
output_file = 'drug_gene_disease_coordinate_normalized.txt'  # 替换为你想要保存的文件路径

# 执行归一化操作
normalize_txt_file(input_file, output_file)