import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. قراءة البيانات (تأكد من المسار الصحيح)
file_path = r"C:\Users\MG\Downloads\Level 3 Codveda\churn-bigml-80.csv"
df = pd.read_csv(file_path)

# 2. تنظيف البيانات (التعامل مع الأعمدة الرقمية فقط)
# الكود ده بيطلع الأعمدة اللي فيها أرقام عشان نتجنب أي خطأ في الأسماء
df_numeric = df.select_dtypes(include=['number'])

# 3. تحديد المدخلات (X) والهدف (y)
# بنفترض إن آخر عمود (الـ Churn) هو اللي عايزين نتوقعه
X = df_numeric.iloc[:, :-1]
y = df.iloc[:, -1]

# 4. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. تدريب الموديل
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6. التوقع
y_pred = model.predict(X_test)

# 7. الطباعة والتقييم
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. رسم الـ Confusion Matrix (الرسمة المطلوبة)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Churn Prediction)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()