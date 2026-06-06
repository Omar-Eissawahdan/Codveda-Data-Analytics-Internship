import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. قراءة الداتا وتخطي السطور المشوهة
df = pd.read_csv('4) house Prediction Data Set.csv', sep=None, engine='python', on_bad_lines='skip')

# 2. تنظيف سريع (Cleaning)
df = df.drop_duplicates()
df = df.fillna(df.mean())

# 3. اختيار الأعمدة الأساسية للتقسيم (عشان الرسمة تطلع واضحة)
# بنختار أول 3 أعمدة كمثال
X = df.iloc[:, :3] 

# 4. تجهيز الأرقام (Scaling) عشان الموديل يشتغل بدقة
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. تطبيق خوارزمية K-Means (تقسيم الداتا لـ 3 مجموعات)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 6. رسم النتيجة
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Cluster'], cmap='viridis')
plt.title('House Data Clustering (K-Means)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

print("تم عمل الـ Clustering بنجاح! الرسمة بتوضح تقسيم البيوت لمجموعات.")