import pandas as pd

# قراءة الداتا الأصلية
df = pd.read_csv('4) house Prediction Data Set.csv', sep=None, engine='python', on_bad_lines='skip')

# 1. شيل أي صفوف مكررة
df = df.drop_duplicates()

# 2. ملء القيم الناقصة بالمتوسط
df = df.fillna(df.mean())

# 3. إزالة الأعمدة التي لا تؤثر (لو افترضنا إنك مش عايز أرقام عشوائية)
# هنختار أعمدة محددة بس عشان الرسمة تطلع واضحة (مثلاً أول 5 أعمدة والسعر)
df_clean = df.iloc[:, [0, 1, 2, 4, 5, -1]] 

# 4. حفظ الداتا النظيفة في ملف جديد
df_clean.to_csv('cleaned_house_data.csv', index=False)

print("تم التنظيف بنجاح! الملف النظيف اسمه: cleaned_house_data.csv");