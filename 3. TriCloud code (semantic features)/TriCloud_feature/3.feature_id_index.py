import pandas as pd
import json
from tqdm import tqdm

# 初始化tqdm的pandas支持
tqdm.pandas()

def load_mapping(file_path):
    """加载JSON映射文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_files():
    # 1. 加载所有映射文件
    print("⏳ 加载映射文件...")
    drug_to_id = load_mapping('./data/drug_to_id.json')
    gene_to_id = load_mapping('./data/gene_to_id.json')
    disease_to_id = load_mapping('./data/disease_to_id.json')

    # 2. 读取原始TSV数据
    print("⏳ 读取TSV文件...")
    df = pd.read_csv('drug_gene_disease_features_negative.tsv', sep='\t')

    # 3. 在原列上直接进行ID映射（带进度条）
    print("⏳ 映射ID...")
    # 只替换三列，其他列保持原样
    df['Drug'] = df['Drug'].progress_apply(lambda x: drug_to_id.get(x.strip(), -1))
    df['Gene'] = df['Gene'].progress_apply(lambda x: gene_to_id.get(x.strip(), -1))
    df['Disease'] = df['Disease'].progress_apply(lambda x: disease_to_id.get(x.strip(), -1))

    # 4. 只对这三列转换数据类型，其他列保持不变
    df[['Drug', 'Gene', 'Disease']] = df[['Drug', 'Gene', 'Disease']].astype(int)

    # 5. 检查未映射的项目
    missing_drugs = df[df['Drug'] == -1]['Drug'].index
    missing_genes = df[df['Gene'] == -1]['Gene'].index
    missing_diseases = df[df['Disease'] == -1]['Disease'].index

    if len(missing_drugs) > 0:
        print(f"⚠️ 未映射的药物数量: {len(missing_drugs)}")
    if len(missing_genes) > 0:
        print(f"⚠️ 未映射的基因数量: {len(missing_genes)} (例如: P08684)")
    if len(missing_diseases) > 0:
        print(f"⚠️ 未映射的疾病数量: {len(missing_diseases)}")

    # 6. 保存结果（保留所有原始列）
    output_file = 'drug_gene_disease_features_negative_mapped.tsv'
    df.to_csv(output_file, sep='\t', index=False)
    print(f"✅ 映射完成，结果已保存至: {output_file}")
    print(f"处理后的数据样例:\n{df.head()}")

if __name__ == "__main__":
    process_files()
