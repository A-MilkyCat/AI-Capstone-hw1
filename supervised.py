import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 讀取 CSV 檔案
df = pd.read_csv("netflix_trailers.csv")

# 定義 Category ID 對應的名稱
def get_category_name(category):
    category_map = {
        1: "Drama",
        2: "Comedy",
        3: "Action/Adventure",
        4: "Sci-Fi/Fantasy",
        5: "Horror/Thriller",
        6: "Documentary",
        7: "Animation"
    }
    return category_map.get(category, "Unknown")

# 過濾出有效的 Category ID
df = df[df["Category ID"].notna()]
df["Category Name"] = df["Category ID"].apply(get_category_name)

# 繪製所有類別的回歸線
plt.figure(figsize=(10, 6))
colors = sns.color_palette("tab10", len(df["Category ID"].unique()))

for i, (cat_id, group) in enumerate(df.groupby("Category ID")):
    X = group[["Duration"]].values
    y = group["Views"].values
    
    if len(X) > 1:  # 確保有足夠數據進行回歸
        # 使用 sklearn 進行線性回歸
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)  # 計算 R^2
        
        # 繪製散點圖
        plt.scatter(X, y, color=colors[i], alpha=0.6, label=f"{get_category_name(cat_id)} (R²={r2:.2f})")
        
        # 繪製回歸線
        X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
        y_range_pred = model.predict(X_range)
        plt.plot(X_range, y_range_pred, color=colors[i], label=f"{get_category_name(cat_id)} Regression")

# 設置標題與標籤
plt.xlabel("Duration (seconds)")
plt.ylabel("Views")
plt.title("Regression Models for Different Categories with R² Values")
plt.legend()
plt.show()

# 繪製不同類別的觀看數箱型圖
plt.figure(figsize=(10, 6))
sns.boxplot(x="Category Name", y="Views", data=df, palette="tab10")
plt.xticks(rotation=45)
plt.title("View Distribution by Category")
plt.show()

# 分析最佳時長區間
df["Duration Group"] = pd.cut(df["Duration"], bins=[0, 60, 120, 180, 240, np.inf], labels=["0-60s", "61-120s", "121-180s", "181-240s", "240+s"])
plt.figure(figsize=(10, 6))
sns.boxplot(x="Duration Group", y="Views", data=df, palette="coolwarm")
plt.title("Views Distribution by Trailer Duration Group")
plt.show()