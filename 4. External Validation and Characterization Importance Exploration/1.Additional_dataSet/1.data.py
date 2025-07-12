import csv

# 读取 node_num.csv 并创建 ID 到名称的映射
id_to_name = {}
with open('node_num.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        name, id = row
        id_to_name[int(id)] = name

# 替换 neg0.csv 中的 ID 为名称
with open('neg0.csv', 'r') as infile, open('1.data/neg0_named.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row in reader:
        drug_id, target_id, disease_id, label = row
        drug_name = id_to_name.get(int(drug_id), drug_id)
        target_name = id_to_name.get(int(target_id), target_id)
        disease_name = id_to_name.get(int(disease_id), disease_id)  # 新增：替换疾病ID
        writer.writerow([drug_name, target_name, disease_name, label])

# 替换 pos.csv 中的 ID 为名称
with open('pos.csv', 'r') as infile, open('1.data/pos_named.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row in reader:
        drug_id, target_id, disease_id, label = row
        drug_name = id_to_name.get(int(drug_id), drug_id)
        target_name = id_to_name.get(int(target_id), target_id)
        disease_name = id_to_name.get(int(disease_id), disease_id)  # 新增：替换疾病ID
        writer.writerow([drug_name, target_name, disease_name, label])