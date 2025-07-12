import numpy as np
import pandas as pd

# 读取txt文件数据函数，跳过首行列名
def read_data(file_path):
    data = pd.read_csv(file_path, delimiter=',', skiprows=1, header=None)
    return data

# 合并两个文件的数据并添加新列
def merge_and_add_column(file1_path, file2_path, output_file):
    # 读取第一个文件的数据
    data1 = read_data(file1_path)
    # 读取第二个文件的数据
    data2 = read_data(file2_path)

    # 检查两文件的行数是否一致
    if len(data1) != len(data2):
        raise ValueError("两个文件的行数不一致，请检查输入文件！")

    # 将两个文件的列横向合并（data1的3列 + data2的3列）
    merged_data = pd.concat([data1, data2], axis=1)

    # 添加新的一列，值全为1
    merged_data[len(merged_data.columns)] = 1

    # 保存结果到新的txt文件，不保存列名
    merged_data.to_csv(output_file, sep=',', header=False, index=False)
    print(f"合并完成并已保存到 {output_file}")

# 输入输出文件路径
file1_path = './drug_gene_disease_coordinate_normalized.txt'  # 替换为你的第一个文件路径
file2_path = './drug_gene_disease_normals.txt' # 替换为你的第二个文件路径
output_file = './drug_gene_disease_3D.txt' # 替换为你想要保存的文件路径

# 执行合并操作
merge_and_add_column(file1_path, file2_path, output_file)