import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. قراءة الداتا
df = pd.read_csv('1) iris.csv').drop_duplicates()

# 2. حساب الإحصائيات (المتوسط، الوسيط، إلخ)
print("الإحصائيات الوصفية للداتا:")
print(df.describe())

# 3. إعداد الرسم
plt.figure(figsize=(10, 6))

# رسم توزيع البيانات (Histogram)
sns.histplot(data=df, x='sepal_length', kde=True, hue='species')
plt.title('Distribution of Sepal Length')
plt.show()

# رسم الـ Boxplot (عشان نشوف القيم الشاذة)
sns.boxplot(data=df, x='species', y='petal_length')
plt.title('Petal Length per Species')
plt.show()

# رسم الـ Scatter plot (عشان نشوف العلاقة بين الأبعاد)
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')
plt.title('Sepal Length vs Sepal Width')
plt.show()