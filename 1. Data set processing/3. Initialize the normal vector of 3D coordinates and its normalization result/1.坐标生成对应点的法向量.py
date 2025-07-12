import numpy as np
from sklearn.neighbors import NearestNeighbors

# 读取txt文件中的数据
def read_data(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # 假设第一行是列名
    return data

# 计算局部平面的法向量
def compute_normal_vector(points):
    # 计算协方差矩阵
    covariance_matrix = np.cov(points, rowvar=False)
    # 对协方差矩阵进行特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # 最小特征值对应的特征向量即为法向量
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
    return normal_vector

# 主函数：计算每个点的法向量
def compute_normals(data, num_neighbors=5):
    # 使用KNN找到每个点的最近邻
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)

    normals = []
    for i in range(len(data)):
        # 获取当前点及其邻居的坐标
        neighbor_points = data[indices[i]]
        # 计算法向量
        normal_vector = compute_normal_vector(neighbor_points)
        normals.append(normal_vector)
    
    return np.array(normals)

# 保存结果到txt文件
def save_normals_to_file(normals, output_file):
    # np.savetxt(output_file, normals, delimiter=',', header='Nx,Ny,Nz', fmt='%.6f', comments='')
     np.savetxt(output_file, normals, delimiter=',', fmt='%.6f', comments='')

# 主程序
if __name__ == "__main__":
    # 输入文件路径和输出文件路径
    input_file = 'drug_gene_disease_coordinate.txt'  # 替换为你的输入文件路径
    output_file = 'drug_gene_disease_coordinate_normals.txt'     # 替换为你的输出文件路径

    # 读取数据
    data = read_data(input_file)

    # 计算法向量
    normals = compute_normals(data, num_neighbors=5)

    # 保存结果
    save_normals_to_file(normals, output_file)

    print("法向量已计算并保存到文件:", output_file)