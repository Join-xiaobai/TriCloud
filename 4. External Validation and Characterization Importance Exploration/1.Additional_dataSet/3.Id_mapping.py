import csv
import json
from tqdm import tqdm

# 加载三个映射文件（请根据你的实际路径修改）
with open('drug_to_id.json', 'r') as f:
    drug_name_to_id = json.load(f)

with open('gene_to_id.json', 'r') as f:
    gene_name_to_id = json.load(f)

with open('disease_to_id.json', 'r') as f:
    disease_name_to_id = json.load(f)

def process_pos_file(input_path, output_path, desc="Processing"):
    with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile, delimiter='\t')

        # 写入新表头
        header = next(reader)
        new_header = ['Drug_ID', 'Gene_ID', 'Disease_ID'] + header[3:]  # Features
        writer.writerow(new_header)

        not_found_count = 0
        rows = list(reader)

        for row in tqdm(rows, desc=desc):
            if len(row) < 4:
                continue

            drug_name, gene_name, disease_name, label = row[0], row[1], row[2], row[3]

            drug_id = drug_name_to_id.get(drug_name)
            gene_id = gene_name_to_id.get(gene_name)
            disease_id = disease_name_to_id.get(disease_name)

            if None in (drug_id, gene_id, disease_id):
                not_found_count += 1
                continue

            writer.writerow([drug_id, gene_id, disease_id] + row[3:])

        print(f"[{desc}] Total unmatched entries: {not_found_count} / {len(rows)}")

def process_neg_file(input_path, output_path, feature_dim=64, desc="Processing"):
    with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile, delimiter='\t')

        # 写入新表头（手动构造）
        new_header = ['Drug_ID', 'Gene_ID', 'Disease_ID'] + [f'feature_{i}' for i in range(feature_dim)]
        writer.writerow(new_header)

        not_found_count = 0
        rows = list(reader)

        for row in tqdm(rows, desc=desc):
            if len(row) < 4:
                continue

            # 输入顺序是: Disease, Drug, Gene, Label（我们不需要 Label）
            disease_name, drug_name, gene_name, label = row

            # 获取 ID
            drug_id = drug_name_to_id.get(drug_name)
            gene_id = gene_name_to_id.get(gene_name)
            disease_id = disease_name_to_id.get(disease_name)

            if None in (drug_id, gene_id, disease_id): #如果没有匹配到实体那么就不保留当前所在行的数据
                not_found_count += 1
                continue

            # 构造特征部分（模拟填充 feature_0 到 feature_63）
            # 如果你没有真实特征，可以填充默认值如 0.0
            features = [0.0] * feature_dim  # 默认填充 0.0，替换为你的真实特征即可

            # 写入结果
            writer.writerow([drug_id, gene_id, disease_id] + features)

        print(f"[{desc}] Total unmatched entries: {not_found_count} / {len(rows)}")

# 处理正样本文件
process_pos_file(
    './2.mapping/pos_with_features.csv',
    './3.Id_mapping/pos_final.csv',
    desc="Processing pos_with_features.csv"
)

# 处理负样本文件
process_neg_file(
    './1.data/neg0_named.csv',
    './3.Id_mapping/neg_final.csv',
    desc="Processing neg_with_features.csv"
)