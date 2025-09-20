import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from toolbox import *
figure_set_chinese()


def run_pca_kmeans(data, n_clusters=3, n_components=2):
    """
    PCA + KMeans 聚类可视化
    -----------------------
    data: 原始数据 (样本数 × 特征数)
    n_clusters: 聚类数
    n_components: 降到几维（2D 或 3D）
    """
    # 1) PCA 降维
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)

    # 2) KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)

    # 3) 可视化
    if n_components == 2:
        plt.figure(figsize=(8,6))
        for k in range(n_clusters):
            plt.scatter(reduced[labels==k,0], reduced[labels==k,1], label=f"类 {k+1}")
        plt.xlabel("主成分1"); plt.ylabel("主成分2")
        plt.title("PCA降维 + KMeans聚类结果")
        plt.legend(); plt.grid(True, ls="--", alpha=0.5); plt.show()
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection="3d")
        for k in range(n_clusters):
            ax.scatter(reduced[labels==k,0], reduced[labels==k,1], reduced[labels==k,2], label=f"类 {k+1}")
        ax.set_xlabel("主成分1"); ax.set_ylabel("主成分2"); ax.set_zlabel("主成分3")
        plt.title("PCA降维 + KMeans聚类结果(3D)")
        plt.legend(); plt.show()

    # 4) 输出主成分解释率
    print("📌 主成分方差解释率：", pca.explained_variance_ratio_)
    print(f"累计解释率 = {pca.explained_variance_ratio_.sum():.3f}")

    return reduced, labels, pca


# ========== 主程序 ==========
if __name__ == "__main__":
    # 模拟数据 (60个用户 × 24小时)
    np.random.seed(42)
    data = []
    for i in range(60):
        if i < 15:  # 工厂型
            base = np.sin(np.linspace(0, np.pi, 24))*80 + 150
        elif i < 30:  # 商场型
            base = np.concatenate([np.ones(12)*80, np.linspace(80,240,12)])
        elif i < 45:  # 家庭型
            base = np.concatenate([np.ones(18)*60, np.linspace(60,200,6)])
        else:  # 办公型
            base = (np.sin(np.linspace(0,2*np.pi,24))*50 + 100)
        noise = np.random.normal(0,10,24)
        data.append(base + noise)
    data = np.array(data)

    # 跑 PCA + 聚类
    reduced, labels, pca = run_pca_kmeans(data, n_clusters=4, n_components=2)

    # 打印每类人数
    print("每一类的样本数：", np.bincount(labels))
