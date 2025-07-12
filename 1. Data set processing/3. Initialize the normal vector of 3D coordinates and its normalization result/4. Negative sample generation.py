import pandas as pd
import numpy as np

# 定义函数读取txt文件
def read_data(file_path):
    # 假设文件是逗号分隔的，跳过第一行（列名）
    data = pd.read_csv(file_path, delimiter=',', skiprows=1, header=None)
    return data

# 定义函数生成负样本
def generate_negative_samples(data, num_negatives):
    """
    生成负样本
    :param data: 正样本数据 (DataFrame)
    :param num_negatives: 需要生成的负样本数量
    :return: 负样本数据 (DataFrame)
    """
    negative_samples = []
    all_drugs = data[0].unique()  # 提取所有药物ID
    all_genes = data[1].unique()  # 提取所有靶标ID
    all_diseases = data[2].unique()  # 提取所有疾病ID
    positive_set = set(tuple(row) for row in data.to_numpy())  # 将正样本转为集合，便于快速查找

    while len(negative_samples) < num_negatives:
        # 随机选择一个正样本作为基础
        sample = data.iloc[np.random.randint(0, len(data))].values
        drug_id, gene_id, disease_id = sample

        # 随机打乱一个或多个元素
        if np.random.rand() < 0.5:  # 50%概率替换 Gene_ID
            gene_id = np.random.choice(all_genes)
        else:  # 50%概率替换 Disease_ID
            disease_id = np.random.choice(all_diseases)

        # 检查生成的样本是否已经在正样本集中
        new_sample = (drug_id, gene_id, disease_id)
        if new_sample not in positive_set:
            negative_samples.append(new_sample)

    # 将负样本转换为 DataFrame
    negative_samples_df = pd.DataFrame(negative_samples, columns=[0, 1, 2])
    return negative_samples_df

# 主函数：生成负样本并保存到文件
def generate_and_save_negatives(input_file, output_file, num_negatives):
    # 读取正样本数据
    data = read_data(input_file)

    # 生成负样本
    negative_samples = generate_negative_samples(data, num_negatives)

    # 保存负样本到新的txt文件
    negative_samples.to_csv(output_file, sep=',', header=False, index=False)
    print(f"负样本已生成并保存到 {output_file}")

# 输入输出文件路径和负样本数量
input_file = 'drug_gene_disease_coordinate.txt'  # 替换为你的正样本文件路径
output_file = 'drug_gene_disease_coordinate_negative.txt'  # 替换为你想要保存的负样本文件路径
num_negatives = 19470  # 替换为你需要生成的负样本数量

# 执行生成负样本操作
generate_and_save_negatives(input_file, output_file, num_negatives)