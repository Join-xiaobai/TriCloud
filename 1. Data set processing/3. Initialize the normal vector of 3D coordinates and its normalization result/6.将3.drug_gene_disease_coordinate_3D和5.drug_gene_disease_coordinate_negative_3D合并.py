def merge_txt_files_vertical(file1, file2, output_file):
    with open(output_file, 'w', encoding='utf-8') as fout:
        # 先写入第一个文件的所有行
        with open(file1, 'r', encoding='utf-8') as f1:
            for line in f1:
                fout.write(line)

        # 再写入第二个文件的所有行
        with open(file2, 'r', encoding='utf-8') as f2:
            for line in f2:
                fout.write(line)

    print(f"合并完成，结果已保存到 {output_file}")

# 使用示例
if __name__ == "__main__":
    file1 = "3.drug_gene_disease_coordinate_3D.txt"
    file2 = "5.drug_gene_disease_coordinate_negative_3D.txt"
    output_file = "drug_gene_disease.txt"

    merge_txt_files_vertical(file1, file2, output_file)