import os
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 固定随机种子
def set_seed(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# =================== Step 1: 模型定义（带特征提取功能）===================
class TransformerClassifier(torch.nn.Module):
    def __init__(self, num_drugs, num_genes, num_diseases, embedding_dim=128, nhead=4, num_layers=2, dropout=0.3):
        super(TransformerClassifier, self).__init__()
        self.drug_emb = torch.nn.Embedding(num_drugs, embedding_dim)
        self.gene_emb = torch.nn.Embedding(num_genes, embedding_dim)
        self.disease_emb = torch.nn.Embedding(num_diseases, embedding_dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim * 3, nhead=nhead, dropout=dropout)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分离特征提取层
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 3, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )

        # 分类层
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

        # 冻结特征提取层参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, drugs, genes, diseases, return_features=False):
        d_emb = self.drug_emb(drugs)
        g_emb = self.gene_emb(genes)
        dis_emb = self.disease_emb(diseases)
        x = torch.cat([d_emb, g_emb, dis_emb], dim=1).unsqueeze(0)  # [1, batch_size, 3*emb_dim]
        x = self.transformer(x).squeeze(0)

        # 提取64维特征
        features = self.feature_extractor(x)

        if return_features:
            return features

        return self.classifier(features).squeeze(1)

    # 添加特征提取方法
    def extract_features(self, drugs, genes, diseases):
        with torch.no_grad():
            return self.forward(drugs, genes, diseases, return_features=True)


# =================== Step 2: 自定义 Dataset ===================
class TripleDataset(Dataset):
    def __init__(self, df, drug_to_idx, gene_to_idx, disease_to_idx):
        self.df = df
        self.drug_to_idx = drug_to_idx
        self.gene_to_idx = gene_to_idx
        self.disease_to_idx = disease_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        drug_idx = self.drug_to_idx[row["Drug"]]
        gene_idx = self.gene_to_idx[row["Gene"]]
        disease_idx = self.disease_to_idx[row["Disease"]]
        return drug_idx, gene_idx, disease_idx


# =================== Step 3: 加载数据和构建映射 ===================
def load_data_and_build_maps(file_path):
    df = pd.read_csv(file_path, sep='\t', header=0, names=["Drug", "Gene", "Disease"])
    all_drugs = list(set(df["Drug"]))
    all_genes = list(set(df["Gene"]))
    all_diseases = list(set(df["Disease"]))
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    disease_to_idx = {d: i for i, d in enumerate(all_diseases)}

    return df, drug_to_idx, gene_to_idx, disease_to_idx


# =================== Step 4: 实例化模型和数据集 ===================
MODEL_PATH = "saved_models/transformer_model_with_disease.pth"
BATCH_SIZE = 512  # 根据你的 GPU 显存调整，越大越快

# 加载训练时使用的映射（确保与训练时一致）
train_df, drug_to_idx, gene_to_idx, disease_to_idx = load_data_and_build_maps("./data/drug_gene_disease.tsv")

# 构建模型
model = TransformerClassifier(
    num_drugs=len(drug_to_idx),
    num_genes=len(gene_to_idx),
    num_diseases=len(disease_to_idx),
    embedding_dim=128,
    nhead=2,
    num_layers=1
).to(device)

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# =================== Step 5: 提取相关数据集特征 ===================
print("提取相关数据集特征...")
pos_df = pd.read_csv("./data/drug_gene_disease.tsv", sep='\t', header=0, names=["Drug", "Gene", "Disease"])
pos_dataset = TripleDataset(pos_df, drug_to_idx, gene_to_idx, disease_to_idx)
pos_loader = DataLoader(pos_dataset, batch_size=BATCH_SIZE, shuffle=False)

pos_features = []
with torch.no_grad():
    for batch in tqdm(pos_loader, desc="提取相关特征"):
        drugs, genes, diseases = map(lambda x: x.to(device), batch)
        features = model(drugs, genes, diseases, return_features=True)
        pos_features.append(features.cpu().numpy())

pos_features = np.vstack(pos_features)

# =================== Step 6: 提取不相关数据集特征 ===================
print("\n提取不相关数据集特征...")
neg_df = pd.read_csv("./data/drug_gene_disease_negative.tsv", sep='\t', header=0,
                     names=["Drug", "Gene", "Disease", "label"])
neg_dataset = TripleDataset(neg_df, drug_to_idx, gene_to_idx, disease_to_idx)
neg_loader = DataLoader(neg_dataset, batch_size=BATCH_SIZE, shuffle=False)

neg_features = []
with torch.no_grad():
    for batch in tqdm(neg_loader, desc="提取不相关特征"):
        drugs, genes, diseases = map(lambda x: x.to(device), batch)
        features = model(drugs, genes, diseases, return_features=True)
        neg_features.append(features.cpu().numpy())

neg_features = np.vstack(neg_features)

# =================== Step 7: 保存结果 ===================
# 保存相关数据集特征
feature_columns = [f"feature_{i}" for i in range(64)]
pos_features_df = pd.DataFrame(pos_features, columns=feature_columns)
pos_final_df = pd.concat([pos_df.reset_index(drop=True), pos_features_df], axis=1)
pos_output_file = "drug_gene_disease_features_positive.tsv"
pos_final_df.to_csv(pos_output_file, sep='\t', index=False)
print(f"\n✅ 相关特征提取完成，已保存至：{pos_output_file}")

# 保存不相关数据集特征
neg_features_df = pd.DataFrame(neg_features, columns=feature_columns)
neg_final_df = pd.concat([neg_df.reset_index(drop=True), neg_features_df], axis=1)
neg_output_file = "drug_gene_disease_features_negative.tsv"
neg_final_df.to_csv(neg_output_file, sep='\t', index=False)
print(f"✅ 不相关特征提取完成，已保存至：{neg_output_file}")
