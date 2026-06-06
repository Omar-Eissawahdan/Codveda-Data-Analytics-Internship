import pandas as pd

# 1. قراءة الملف
df = pd.read_csv('1) iris.csv')

# 2. عرض أول 5 سطور عشان نشوف الداتا
print("أول 5 سطور من الداتا:")
print(df.head())

# 3. التأكد هل فيه قيم ناقصة (Missing Values)؟
print("\nعدد القيم الناقصة في كل عمود:")
print(df.isnull().sum())

# 4. التأكد هل فيه صفوف متكررة؟
print("\nعدد الصفوف المتكررة:")
print(df.duplicated().sum())

# 5. لو فيه تكرار (بنزيله) - ده الكود الاحتياطي
df = df.drop_duplicates()

# 6. معلومات عامة عن الداتا (عشان نتأكد إن كل حاجة تمام)
print("\nمعلومات عن الداتا:")
print(df.info())