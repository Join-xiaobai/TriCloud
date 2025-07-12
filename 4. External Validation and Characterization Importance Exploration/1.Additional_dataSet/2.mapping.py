import csv
from tqdm import tqdm

# Step 1: 读取正样本特征 TSV 文件，并建立 (Drug, Gene, Disease) -> 特征列表 的映射
positive_features = {}
with open('drug_gene_disease_features_positive.tsv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)  # 跳过表头
    for row in reader:
        drug, gene, disease = row[0], row[1], row[2]
        key = (drug, gene, disease)
        positive_features[key] = row[3:]  # 只保留特征部分

# Step 2: 读取负样本特征 TSV 文件
negative_features = {}
with open('drug_gene_disease_features_negative.tsv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)  # 跳过表头
    for row in reader:
        drug, gene, disease = row[0], row[1], row[2]
        key = (drug, gene, disease)
        negative_features[key] = row[3:]

# Step 3: 处理 pos_named.csv 并匹配特征
with open('./1.data/pos_named.csv', 'r', newline='', encoding='utf-8') as infile, \
     open('./2.mapping/pos_with_features.csv', 'w', newline='') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # 写入新表头
    # new_header = ['Drug', 'Gene', 'Disease', 'Label'] + [f'feature_{i}' for i in range(64)]
    new_header = ['Drug', 'Gene', 'Disease'] + [f'feature_{i}' for i in range(64)]
    writer.writerow(new_header)

    rows = list(reader)  # 一次性读取所有行以便显示进度条
    not_found_count = 0

    for row in tqdm(rows, desc="Processing pos_named.csv", total=len(rows)):
        if len(row) < 4:
            print(f"[Pos Named File] Skipping invalid row: {row}")
            not_found_count += 1
            continue

        drug, disease, gene, label = row  # 注意：原文件中是 Drug, Disease, Gene
        key = (drug, gene, disease)  # 调整为 Drug, Gene, Disease 才能匹配特征文件
        features = positive_features.get(key)
        if features:
            # writer.writerow([drug, gene, disease, label] + features)
            writer.writerow([drug, gene, disease] + features)
        else:
            not_found_count += 1

    print(f"[Pos] Total unmatched entries: {not_found_count} / {len(rows)}")

# Step 4: 处理 neg_named.csv 并匹配特征
# with open('./1.data/neg0_named.csv', 'r', newline='', encoding='utf-8') as infile, \
#      open('./2.mapping/neg_with_features.csv', 'w', newline='') as outfile:
#
#     reader = csv.reader(infile)
#     writer = csv.writer(outfile)
#
#     # 写入新表头
#     new_header = ['Drug', 'Gene', 'Disease', 'Label'] + [f'feature_{i}' for i in range(64)]
#     writer.writerow(new_header)
#
#     rows = list(reader)  # 一次性读取所有行以便显示进度条
#     not_found_count = 0
#
#     for row in tqdm(rows, desc="Processing neg0_named.csv", total=len(rows)):
#         if len(row) < 4:
#             print(f"[Neg Named File] Skipping invalid row: {row}")
#             not_found_count += 1
#             continue
#
#         drug, disease, gene, label = row  # 原文件是 Drug, Disease, Gene
#         key = (drug, gene, disease)  # 调整为 Drug, Gene, Disease
#         features = negative_features.get(key)
#         if features:
#             writer.writerow([drug, gene, disease, label] + features)
#         else:
#             not_found_count += 1
#
#     print(f"[Neg] Total unmatched entries: {not_found_count} / {len(rows)}")