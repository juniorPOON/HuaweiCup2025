import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from toolbox import *
figure_set_chinese()

def find_best_k(data, k_range=(2,8)):
    inertias, silhouettes = [], []
    for k in range(k_range[0], k_range[1]+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)  # n_init=10更稳
        labels = km.labels_
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(data, labels))

    Ks = list(range(k_range[0], k_range[1]+1))
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(Ks, inertias, "o-", label="SSE(肘部法则)")
    ax1.set_xlabel("聚类数 K"); ax1.set_ylabel("SSE", color="b"); ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ax2.plot(Ks, silhouettes, "s--r", label="轮廓系数")
    ax2.set_ylabel("轮廓系数", color="r"); ax2.tick_params(axis="y", labelcolor="r")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.title("最佳聚类数选择：肘部法则 & 轮廓系数"); plt.grid(True, ls="--", alpha=.5); plt.show()

    best_k = Ks[int(np.argmax(silhouettes))]
    print(f"📌 建议的最佳 K = {best_k}，对应轮廓系数 = {max(silhouettes):.3f}")
    return best_k

if __name__ == "__main__":
    # —— 这里定义 data ——（45个用户 × 24小时）
    np.random.seed(42)
    data = []

    for i in range(60):
        if i < 15:  # 工厂型（白天高，晚上低）
            base = np.sin(np.linspace(0, np.pi, 24)) * 80 + 150
        elif i < 30:  # 商场型（下午和晚上高）
            base = np.concatenate([np.ones(12) * 80, np.linspace(80, 240, 12)])
        elif i < 45:  # 家庭型（晚间高峰）
            base = np.concatenate([np.ones(18) * 60, np.linspace(60, 200, 6)])
        else:  # 办公型（上午和下午高，中午和晚上低）
            base = (np.sin(np.linspace(0, 2 * np.pi, 24)) * 50 + 100)
        # 加噪声（让类别更难分）
        noise = np.random.normal(0, 10, 24)
        data.append(base + noise)

    # 转成 numpy 数组：60 个用户 × 24 小时
    data = np.array(data)

    best_k = find_best_k(data, k_range=(2,6))

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(data)
    print("每一类样本数：", np.bincount(labels))
