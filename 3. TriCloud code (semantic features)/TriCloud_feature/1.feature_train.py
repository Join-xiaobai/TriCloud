import os
import logging
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

def set_seed(seed=42):
    """
    设置所有相关模块的随机种子，以保证结果的可重复性。

    :param seed: 随机种子，默认值为 42
    """
    # Python 内置随机模块
    random.seed(seed)

    # Numpy 随机种子
    np.random.seed(seed)

    # PyTorch 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用 GPU

# 设置随机种子
set_seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 日志设置
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/train_with_disease.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =================== Step 1: 加载数据并构建正样本 ===================
def load_positive_samples(file_path):
    df = pd.read_csv(file_path, sep='\s+', header=0, names=["Drug", "Gene", "Disease"])
    # 构建所有唯一的 Drug-Gene-Disease 组合（正样本）
    positive_triples = set(zip(df["Drug"], df["Gene"], df["Disease"]))
    return df, positive_triples

# =================== Step 2: 生成负样本 ===================
def generate_negative_samples(positive_triples, all_drugs, all_genes, all_diseases, num_neg_per_pos=1):
    negative_samples = []
    existing_pos = set(positive_triples)
    for drug, gene, disease in tqdm(existing_pos, desc="Generating negative samples"):
        for _ in range(num_neg_per_pos):
            while True:
                neg_drug = random.choice(all_drugs)
                neg_gene = random.choice(all_genes)
                neg_disease = random.choice(all_diseases)
                if (neg_drug, neg_gene, neg_disease) not in existing_pos:
                    negative_samples.append((neg_drug, neg_gene, neg_disease))
                    break
    return negative_samples

# =================== Step 3: 创建 Dataset 类 ===================
class TripleDataset(Dataset):
    def __init__(self, pos_samples, neg_samples, drug_to_idx, gene_to_idx, disease_to_idx):
        self.data = [(d, g, dis, 1) for d, g, dis in pos_samples] + \
                    [(d, g, dis, 0) for d, g, dis in neg_samples]
        self.drug_to_idx = drug_to_idx
        self.gene_to_idx = gene_to_idx
        self.disease_to_idx = disease_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        drug, gene, disease, label = self.data[idx]
        return (
            self.drug_to_idx[drug],
            self.gene_to_idx[gene],
            self.disease_to_idx[disease],
            label
        )

# =================== Step 4: 构建 Transformer 模型 ===================
class TransformerClassifier(nn.Module):
    def __init__(self, num_drugs, num_genes, num_diseases, embedding_dim=128, nhead=4, num_layers=2, dropout=0.5):
        super(TransformerClassifier, self).__init__()
        self.drug_emb = nn.Embedding(num_drugs, embedding_dim)
        self.gene_emb = nn.Embedding(num_genes, embedding_dim)
        self.disease_emb = nn.Embedding(num_diseases, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim * 3, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分离特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
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


# =================== Step 5: 训练与验证函数 ===================
def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for drugs, genes, diseases, labels in dataloader:
            drugs, genes, diseases, labels = drugs.to(device), genes.to(device), diseases.to(device), labels.float().to(device)
            outputs = model(drugs, genes, diseases)
            preds = (outputs > 0.5).float()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)

    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "aupr": aupr
    }

def train():
    print("Loading data...")
    df, positive_triples = load_positive_samples("./data/drug_gene_disease.tsv")
    all_drugs = list(set(df["Drug"]))
    all_genes = list(set(df["Gene"]))
    all_diseases = list(set(df["Disease"]))

    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    disease_to_idx = {d: i for i, d in enumerate(all_diseases)}

    print(f"Positive triples count: {len(positive_triples)}")
    print("Generating negative samples...")
    negative_triples = generate_negative_samples(positive_triples, all_drugs, all_genes, all_diseases, num_neg_per_pos=1)

    print("Building dataset...")
    dataset = TripleDataset(positive_triples, negative_triples, drug_to_idx, gene_to_idx, disease_to_idx)

    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128)

    model = TransformerClassifier(
        num_drugs=len(all_drugs),
        num_genes=len(all_genes),
        num_diseases=len(all_diseases),
        embedding_dim=128,
        nhead=2,
        num_layers=1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    best_val_auc = 0.0
    epochs = 200

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for drugs, genes, diseases, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            drugs, genes, diseases, labels = drugs.to(device), genes.to(device), diseases.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(drugs, genes, diseases)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        metrics = evaluate(model, val_loader)
        log_msg = (
            f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | "
            f"Val ACC: {metrics['acc']:.4f}, F1: {metrics['f1']:.4f}, "
            f"AUC: {metrics['auc']:.4f}, AUPR: {metrics['aupr']:.4f}"
        )
        print(log_msg)
        logging.info(log_msg)

        # Save best model
        if metrics['auc'] > best_val_auc:
            best_val_auc = metrics['auc']
            os.makedirs("saved_models", exist_ok=True)
            torch.save(model.state_dict(), "saved_models/transformer_model_with_disease.pth")
            print("Model saved!")

    print("Training finished.")

if __name__ == "__main__":
    train()