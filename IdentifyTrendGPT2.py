from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def train_model(stock_data):
    # Przygotowanie danych
    df = pd.DataFrame(stock_data)
    # Inżynieria cech,
    X = df[['open', 'high', 'low', 'close', 'volume']]
    #  'trend' to kolumna z etykietami klas, w tym przypadku -1 (spadkowy), 0 (boczny, 1 (rosnąc
    y = df['trend']

    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Uczenie modelu
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Zwrócenie wytrenowanego modelu
    return model

def predict_trend(model, recent_data):
    # Przygotowanie danych dla ostatnich 20 dni
    recent_df = pd.DataFrame(recent_data)
    X_recent = recent_df[['open', 'high', 'low', 'close', 'volume']]

    # Przewidywanie trendu
    predicted_trend = model.predict(X_recent)

    return predicted_trend


def aggregate_trend(predicted_trend):
    count_rise = sum(predicted_trend == 1)
    count_fall = sum(predicted_trend == -1)

    if count_rise > count_fall:
        return 1  # Trend wzrostowy
    elif count_fall > count_rise:
        return -1  # Trend spadkowy
    else:
        return 0  # Trend boczny

def add_trend_label(stock_data):
    # Konwertowanie danych do obiektu DataFrame
    df = pd.DataFrame(stock_data)

    # Sortowanie danych po dacie w kolejności rosnącej
    df.sort_values(by='date', inplace=True)

    # Obliczenie SMA-20
    df['SMA_20'] = df['close'].rolling(window=20).mean()

    # tworzenie kolumny  'dx' i 'dy', dzięki którym będzie można obliczyć kąt nachylenie w wykresie
    df['y_index'] = df.index
    df['dx'] = df['y_index'].diff()
    df['dy'] = df['SMA_20'].diff()

    # Obliczanie kątu nachylenia w radianach
    df['angle'] = np.arctan2(df['dy'], df['dx'])
    df['trend'] = df['angle'].apply(lambda x: 1 if x > 0.01 else (-1 if x < -0.01 else 0))

    return df.to_dict(orient='records')


#0 wczytywanie danych z pliku .csv
fileName = 'cbf_d.csv'
data = pd.read_csv(fileName)

data = data.rename(columns={'Data': 'date', 'Otwarcie': 'open','Najwyzszy':'high','Najnizszy':'low','Zamkniecie':'close', 'Wolumen':'volume'})

#czyszcenie danych
data.dropna(inplace=True)

#1 Dodanie etykiety "trend" do danych giełdowych
stock_data_with_trend = add_trend_label(data)
print(stock_data_with_trend)

#2 Trenowanie modelu
trained_model = train_model(stock_data_with_trend)

#3 Przewidywanie trendu
recent_data = data.tail(20)

predicted_trend = predict_trend(trained_model, recent_data)
print("Przewidywany trend:", predicted_trend)

#4 agregiacja trendu
overall_trend = aggregate_trend(predicted_trend)
print("Całkowity trend w ostatnich 20 dniach:", fileName, overall_trend)






from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

def classify_stock_trend_last_20_days(data):
    # Convert data to a pandas DataFrame
    df = pd.DataFrame(data)

    # Calculate the percentage change in the closing price
    df['Close_pct_change'] = df['close'].pct_change()

    # Determine the trend based on the percentage change
    df['Trend'] = df['Close_pct_change'].apply(lambda x: 1 if x > 0.015 else (-1 if x < -0.015 else 0))

    # Prepare data for classification
    X = df[['close', 'Close_pct_change']]
    y = df['Trend']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the classifier
    knn.fit(X_scaled, y)

    # Predict the trend for the last 20 days
    last_20_days_data = df.tail(20)
    X_last_20_days = last_20_days_data[['close', 'Close_pct_change']]
    X_last_20_days_imputed = imputer.transform(X_last_20_days)
    X_last_20_days_scaled = scaler.transform(X_last_20_days_imputed)
    predicted_trend = knn.predict(X_last_20_days_scaled)

    return predicted_trend

#print('xd',classify_stock_trend_last_20_days(data))