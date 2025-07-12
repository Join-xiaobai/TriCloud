import pandas as pd
import json

# 读取文件
file1_path = 'E:\\Users\\Administrator\\bioinfo\\pointnet.pytorch-master\\dataset\\dataprocess\\ChG-Miner_miner-chem-gene.tsv'
file2_path = 'E:\\Users\\Administrator\\bioinfo\\pointnet.pytorch-master\\dataset\\dataprocess\\DCh-Miner_miner-disease-chemical.tsv'
file3_path = 'E:\\Users\\Administrator\\bioinfo\\pointnet.pytorch-master\\dataset\\dataprocess\\DG-Miner_miner-disease-gene.tsv'

df1 = pd.read_csv(file1_path, sep='\t')  # Drug 和 Gene
df2 = pd.read_csv(file2_path, sep='\t')  # Disease(MESH) 和 Chemical
df3 = pd.read_csv(file3_path, sep='\t')  # Disease(MESH) 和 Gene

# 加载编号字典
with open('drug_to_id.json', 'r') as f:
    drug_to_id = json.load(f)
with open('gene_to_id.json', 'r') as f:
    gene_to_id = json.load(f)
with open('disease_to_id.json', 'r') as f:
    disease_to_id = json.load(f)

# 构建疾病与靶标的关系
disease_gene_dict = df3.set_index('# Disease(MESH)')['Gene'].to_dict()

# 构建药物-靶标-疾病关系
drug_gene_disease = []
for _, row in df1.iterrows():
    drug = row['#Drug']
    gene = row['Gene']
    # 查找与靶标相关的疾病
    diseases = [disease for disease, g in disease_gene_dict.items() if g == gene]
    for disease in diseases:
        drug_gene_disease.append((drug, gene, disease))

# 替换为编号
drug_gene_disease_mapped = []
for drug, gene, disease in drug_gene_disease:
    drug_id = drug_to_id.get(drug, -1)  # 如果找不到，返回 -1
    gene_id = gene_to_id.get(gene, -1)
    disease_id = disease_to_id.get(disease, -1)
    if drug_id != -1 and gene_id != -1 and disease_id != -1:
        drug_gene_disease_mapped.append((drug_id, gene_id, disease_id))

# 保存结果
result_df = pd.DataFrame(drug_gene_disease_mapped, columns=['Drug_ID', 'Gene_ID', 'Disease_ID'])
result_df.to_csv('drug_gene_disease_mapping.tsv', sep='\t', index=False)

print("药物-靶标-疾病关系文件已保存为 drug_gene_disease_mapping.tsv")
def tsv_to_txt():
    # 读取 TSV 文件
    tsv_file_path = 'drug_gene_disease_mapping.tsv'
    df = pd.read_csv(tsv_file_path, sep='\t')

    # 将数据写入 TXT 文件
    txt_file_path = 'drug_gene_disease_mapping.txt'
    with open(txt_file_path, 'w') as txt_file:
        # 写入列名（去掉空格）
        txt_file.write(','.join(df.columns) + '\n')
        # 写入数据（去掉空格）
        for _, row in df.iterrows():
            txt_file.write(','.join(map(str, row)) + '\n')

    print(f"文件已保存为 {txt_file_path}")

if __name__ == '__main__':
    tsv_to_txt()