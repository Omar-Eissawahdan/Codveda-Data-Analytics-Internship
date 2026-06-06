import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# 1. قراءة الملف
# (بنتأكد إن الـ sep مظبوط حسب نوع ملفك)
df = pd.read_csv('3) Sentiment dataset.csv', on_bad_lines='skip')

# 2. تنظيف بسيط (بنشيل أي تعليق فاضي)
df = df.dropna()

# 3. وظيفة صغيرة عشان نعرف النص إيجابي ولا سلبي
def get_sentiment(text):
    analysis = TextBlob(str(text))
    # polarity بترجع رقم من -1 (سلبي خالص) لـ 1 (إيجابي خالص)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# 4. تطبيق الوظيفة على كل الكومنتات
# (افترضنا إن اسم العمود اللي فيه الكلام هو 'text'، لو اسم عمود تاني غيره في الكود)
# لو مش عارف اسم العمود، اطبع df.columns قبل السطر ده
df['Sentiment'] = df.iloc[:, 0].apply(get_sentiment)

# 5. عرض النتائج
print(df.head())

# 6. رسم بياني يوريك كام واحد إيجابي وكام واحد سلبي
df['Sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Sentiment Analysis Results')
plt.show()