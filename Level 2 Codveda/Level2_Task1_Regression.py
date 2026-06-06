import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# اقرأ الداتا النظيفة
df = pd.read_csv('cleaned_house_data.csv')

# ارسم الـ Heatmap الجديدة
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Cleaned Data Correlation (Important Factors Only)')
plt.show()