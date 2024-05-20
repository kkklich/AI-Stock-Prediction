import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def read_csv_file(fileName):
    stock_data = pd.read_csv(fileName)
    data = pd.DataFrame(stock_data)

    data = data.rename(
        columns={'Data': 'date', 'Otwarcie': 'open', 'Najwyzszy': 'high', 'Najnizszy': 'low', 'Zamkniecie': 'close',
                 'Wolumen': 'volume'})
    # czyszcenie danych
    data.dropna(inplace=True)

    return data


data = read_csv_file('files/cmr_d.csv')
data['trend'] = data['close'].diff(periods=20).apply(lambda x: 1 if x > 0 else 0)

print(data)


def create_features(data, window=20):
    features = []
    for i in range(len(data) - window):
        features.append(data['close'].iloc[i:i+window].values)
    return np.array(features)

X = create_features(data, window=20)
y = data['trend'].iloc[20:].values

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

last_1500_rows = data.tail(20)

recent_df = pd.DataFrame(last_1500_rows)
print(recent_df)
new_data = recent_df[['date', 'close']]

#prediction = model.predict(X_recent)
#print(prediction)


#new_data = {
  #  'date': pd.date_range(start='2023-04-11', periods=20),
 #   'close': np.random.rand(20) * 100
#}
#new_df = pd.DataFrame(new_data)
new_features = new_data['close'].values.reshape(1, -1)

trend_prediction = model.predict(new_features)
print('Trend:', 'Wzrostowy' if trend_prediction == 1 else 'Spadkowy')