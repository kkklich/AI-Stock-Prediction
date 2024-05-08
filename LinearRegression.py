import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Wczytanie danych
data = pd.read_csv('pkn_d.csv')

data = data.rename(columns={'Data': 'Date', 'Otwarcie': 'Open','Najwyzszy':'High','Najnizszy':'Low','Zamkniecie':'Close', 'Wolumen':'Volume'})



window_size = 10
data['rolling_mean'] = data['Close'].rolling(window=window_size).mean()
data.dropna(inplace=True)
print(data)

#data.drop('Open', axis=1, inplace=True)
#data.drop('High', axis=1, inplace=True)
#data.drop('Low', axis=1, inplace=True)
#data.drop('Close', axis=1, inplace=True)

# Displaying the DataFrame after removing the column
print(data)

# Przygotowanie danych
#X = data[['Open', 'High', 'Low', 'Volume', 'rolling_mean']]  # Możesz dostosować cechy do analizy
X = data[['Close','rolling_mean','Volume']]  # Możesz dostosować cechy do analizy
y = data.index
#y = data['Date']

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utworzenie i trenowanie modelu regresji liniowej
model = LinearRegression()
model.fit(X_train, y_train)

# Przewidywanie cen akcji dla następnych 20 dni na podstawie ostatnich danych
last_data = X_test.tail(1)  # Ostatnie dane testowe
future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=20, freq='D')  # Daty dla przyszłych 20 dni
future_data = pd.DataFrame(columns=X.columns, index=future_dates)
future_data.iloc[0] = last_data.values[0]

print(future_data)

for i in range(1, len(future_data)):
    future_data.iloc[i] = model.predict([future_data.iloc[i-1]])

# Wyświetlenie przewidywanych cen akcji dla następnych 20 dni
print("Przewidywane ceny akcji dla następnych 20 dni:")
print(future_data)



plt.plot(future_data.index, future_data['Close'])

# Setting labels and title
plt.xlabel('Date')
plt.ylabel('Close')
plt.title('High Prices Over Time')

# Rotating x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Displaying the plot
plt.show()