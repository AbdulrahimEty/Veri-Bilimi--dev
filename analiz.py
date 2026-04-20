import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv('data/boston.csv')




df.rename(columns={
    'CRIM'    : 'Suç_Oranı',
    'ZN'      : 'İmar_Oranı',
    'INDUS'   : 'İşletme_Oranı',
    'CHAS'    : 'Nehir',
    'NOX'     : 'Nitrik_Oksit',
    'RM'      : 'Oda_Sayısı',
    'AGE'     : 'Yaş',
    'DIS'     : 'Mesafe',
    'RAD'     : 'Radyal_Otoyol',
    'TAX'     : 'Vergi',
    'PTRATIO' : 'Öğretmen_Oranı',
    'B'       : 'Siyahi_Nüfus',
    'LSTAT'   : 'Düşük_Gelir_Oranı',
    'MEDV'    : 'Ev_Fiyatı'
    }, inplace=True)

#print(df.columns)
df = df[df['Ev_Fiyatı'] < 50]
#print(df.isnull().sum())

df.drop('İmar_Oranı', axis=1, inplace=True)
df.drop('Nehir', axis=1, inplace=True)

#print(df.columns)

plt.figure(figsize=(10,5))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Matrix')
#plt.show()

def correlation_for_dropping(df, threshold):
    columns_to_drop = set()
    corr = df.corr()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                columns_to_drop.add(corr.columns[i])

    return columns_to_drop





#print(correlation_for_dropping(X_train, threshold=0.80))

X = df.drop('Ev_Fiyatı', axis=1) # Metrekare
y = df['Ev_Fiyatı']   # Fiyat


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.drop('Vergi', axis=1, inplace=True)
X_test.drop('Vergi', axis=1, inplace=True)


model = LinearRegression()
model.fit(X_train, y_train)


tahmin = model.predict(X_test)

print(f"R2 Skoru: {r2_score(y_test, tahmin)}")


plt.figure(figsize=(8, 5))
sns.boxplot(y=df['Ev_Fiyatı'], color='skyblue')
plt.title('Ev Fiyatı Dağılımı ve Aykırı Değerler')
plt.show()