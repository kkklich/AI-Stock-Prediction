import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

# Wczytaj dane historyczne cen akcji
data = pd.read_csv('files/cps_d.csv')

data = data.rename(columns={'Data': 'Date', 'Otwarcie': 'Open','Najwyzszy':'High','Najnizszy':'Low','Zamkniecie':'Close', 'Wolumen':'Volume'})
#print(data)

# Inżynieria cech: Dodaj średnią kroczącą jako cechę
#window_size = 10
#data['rolling_mean'] = data['Close'].rolling(window=window_size).mean()
# Usuń wiersze z brakującymi wartościami
data.dropna(inplace=True)
print(data)
print(len( data))
totalLength = len(data)
print('------')


def identify_trend(data):
    # Konwertowanie danych do ramki danych pandas
    df = pd.DataFrame(data)

    # Tworzenie cechy X (numer dni)
    df['Day'] = range(1, len(df) + 1)
    X = df[['Day']]

    # Tworzenie zmiennej docelowej y (cena zamknięcia)
    y = df['Close']

    # Inicjalizacja modelu regresji liniowej
    model = LinearRegression()

    # Trenowanie modelu na danych
    model.fit(X, y)

    # Przewidywanie ceny zamknięcia dla ostatniego dnia
    last_day = len(df)
    last_close_pred = model.predict([[last_day]])[0]

    # Przewidywanie trendu na podstawie prognozy ceny zamknięcia
    last_close = df['Close'].iloc[-1]
    if last_close_pred > last_close:
        trend = 'Bullish upward trend'
    elif last_close_pred < last_close:
        trend = 'Bearish downward trend'
    else:
        trend = 'Neutral'

    print(trend)

    plt.figure(figsize=(10, 6))
    plt.scatter(df['Day'], df['Close'], color='blue', label='Actual Close Price')
    plt.plot(df['Day'], model.predict(X), color='red', label='Linear Regression')
    plt.title('Linear Regression Trend')
    plt.xlabel('Day')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    return trend


def identify_trendClassification(data):
    # Konwertowanie danych do ramki danych pandas
    df = pd.DataFrame(data)

    # Obliczanie zmiany procentowej ceny zamknięcia
    df['Close_pct_change'] = df['Close'].pct_change()

    # Określenie trendu na podstawie zmiany procentowej
    df['Trend'] = df['Close_pct_change'].apply(lambda x: 'Bullish up' if x > 0 else ('Bearish down' if x < 0 else 'Neutral'))

    # Przygotowanie danych do klasyfikacji
    X = df[['Close', 'Close_pct_change']]
    y = df['Trend']

    # Imputacja brakujących wartości
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Standaryzacja danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Inicjalizacja klasyfikatora KNN
    knn = KNeighborsClassifier(n_neighbors=3)

    # Trenowanie klasyfikatora
    knn.fit(X_scaled, y)

    # Przewidywanie trendu dla ostatnich danych
    last_close = df['Close'].iloc[-1]
    last_close_pct_change = df['Close_pct_change'].iloc[-1]
    last_data_imputed = imputer.transform([[last_close, last_close_pct_change]])
    last_data_scaled = scaler.transform(last_data_imputed)
    predicted_trend = knn.predict(last_data_scaled)[0]

    # Rysowanie wykresu
    plt.figure(figsize=(10, 6))
    plt.plot(df['Close'], label='Close Price')
    #print(len(df) - 1, last_close)
    #plt.scatter(len(df) - 1, last_close, color='red', marker='o', label='Last Close Price')
    plt.title('Stock Price and Predicted Trend')
    plt.xlabel('Day')
    plt.ylabel('Close Price')
    plt.axhline(y=last_close, color='gray', linestyle='--')
    plt.text(len(df) - 1, last_close, f'{predicted_trend} Trend', ha='right', va='bottom')
    plt.legend()
    plt.grid(True)
    plt.show()

    return predicted_trend


# Calling the function with the example stock data
last_1500_rows = data.tail(450)
#last_1500_rows = data

#predicted_trend = identify_trend(last_1500_rows)
#print("regresja ",predicted_trend)

trend = identify_trendClassification(last_1500_rows)
print("Obecny trend: KNN", trend)




