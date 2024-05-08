import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Wczytaj dane historyczne cen akcji
data = pd.read_csv('pkn_d.csv')

data = data.rename(columns={'Data': 'Date', 'Otwarcie': 'Open','Najwyzszy':'High','Najnizszy':'Low','Zamkniecie':'Close', 'Wolumen':'Volume'})
print(data)


# Inżynieria cech: Dodaj średnią kroczącą jako cechę
window_size = 10
data['rolling_mean'] = data['Close'].rolling(window=window_size).mean()
# Usuń wiersze z brakującymi wartościami
data.dropna(inplace=True)
print(data)
print('------')
# Podziel dane na cechy (X) i target (y)
#X = data[['Close', 'Volume']].values
X = data[['Close']].values
y = data['rolling_mean'].values

print(X, len(X))
print(y, len(y))

# Skalowanie cech do przedziału [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(X_train, X_test, y_train, y_test )
# Tworzenie modelu sztucznej sieci neuronowej
model = tf.keras.models.Sequential([
   tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
   tf.keras.layers.Dense(64, activation='relu'),
   tf.keras.layers.Dense(1)
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='mean_squared_error')

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Ocena wydajności modelu na danych testowych
loss = model.evaluate(X_test, y_test)
print("RMSE:", np.sqrt(loss))

# Wykres błędu treningowego i walidacyjnego w zależności od epoki
#plt.plot(history.history['loss'], label='train_loss')
##plt.plot(history.history['val_loss'], label='val_loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()

## Przewidywanie trendów cen akcji
predictions = model.predict(X_test)

# Wykres rzeczywistych i przewidywanych cen akcji
#plt.plot(y_test, label='Actual Price')
#plt.plot(predictions, label='Predicted Price')
#plt.xlabel('Time')
#plt.ylabel('Price')
#plt.legend()
#plt.show()


y_test_subset = y_test[:100]
predictions_subset = predictions[:100]

plt.plot(y_test_subset, label='Actual Price')
plt.plot(predictions_subset, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


future = data[:100]
#future_predictions = model.predict()

# Wykres rzeczywistych i przewidywanych cen akcji
#plt.plot(y_test, label='Actual Price')
#plt.plot(range(len(y_test), len(y_test) + len(future_predictions)), future_predictions, label='Predicted Future Price')
#plt.xlabel('Time')
#plt.ylabel('Price')
#plt.title('Actual vs. Predicted Stock Prices')
#plt.legend()
#plt.grid(True)
#plt.show()


