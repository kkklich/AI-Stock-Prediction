import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

selected_columns = ['open', 'close', 'volume', 'SMA_20', 'angle','EMA_50', 'daily_changes','open-close-changes','RSI']

def standarize_data(df):
    # Standardize the data using the MinMaxScaler

    data_scaler = df.loc[:, selected_columns]
    #scaler = MinMaxScaler()   #88.6%
    scaler = StandardScaler()  #90.5%
    df_scaled = scaler.fit_transform(data_scaler)

    return df_scaled

def train_model(stock_data):
    # Przygotowanie danych
    df = pd.DataFrame(stock_data)
    # Inżynieria cech
    df.dropna(inplace=True)
    df_x = df.loc[:, selected_columns]

    X = standarize_data(df_x)
    #  'trend' to kolumna z etykietami klas, w tym przypadku -1 (spadkowy), 0 (boczny, 1 (rosnąc
    y = df['trend']

    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Uczenie modelu regresją logistyczna 0 | 1
    model = LogisticRegression(max_iter=100)

    #model = LinearRegression()
    #model = DecisionTreeClassifier(max_depth=1000)

    model.fit(X_train, y_train)

    #obliczanie dokłądności modelu
    accuracy = model.score(X_test, y_test)
    print("dokładność modelu:", accuracy)

    # Zwrócenie wytrenowanego modelu
    return model

def predict_trend(model, recent_data):
    recent_df = pd.DataFrame(recent_data)
    X_recent = recent_df.loc[:, selected_columns]
    #standaryzacja danych do predykcji
    X_recent = standarize_data(X_recent)

    # Przewidywanie trendu
    predicted_trend = model.predict(X_recent)

    return predicted_trend


def aggregate_trend(predicted_trend):
    count_rise = sum(predicted_trend == 1)
    count_fall = sum(predicted_trend == -1)

    trend_double = sum(predicted_trend) / len(predicted_trend)
    print('prawdopodobieństwo trendu: ', trend_double)

    if count_rise > count_fall:
        return 1  # Trend wzrostowy
    elif count_fall > count_rise:
        return -1  # Trend spadkowy
    else:
        return 0  # Trend boczny


#obliczanie RSI
def calculate_rsi(prices, period):
    # Calculate the difference between each consecutive price
    price_diff = prices.diff()

    # Calculate the gain and loss
    gain = price_diff.where(price_diff > 0, 0)
    loss = -price_diff.where(price_diff < 0, 0)

    # Calculate the average gain and average loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Calculate the relative strength
    relative_strength = avg_gain / avg_loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + relative_strength))

    return rsi


#feature engineering tworzenie cech
def add_trend_label(stock_data, SMA_window=20):
    # Konwertowanie danych do obiektu DataFrame
    df = pd.DataFrame(stock_data)

    # Obliczenie cech
    df['SMA_20'] = df['close'].rolling(window=SMA_window).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['EMA_50'] =  df['close'].ewm(span=5, adjust=False).mean()
    df['daily_changes'] = df['close'].pct_change()
    df['open-close-changes'] =  (df['high'] - df['low']) / df['high']
    df['RSI'] = calculate_rsi(df['close'], period=3)

    # tworzenie kolumny  'dx' i 'dy', dzięki którym będzie można obliczyć kąt nachylenie w wykresie
    df['y_index'] = df.index
    df['dx'] = df['y_index'].diff()
    df['dy'] = df['SMA_20'].diff()
    # Obliczanie kątu nachylenia w radianach
    df['angle'] = np.arctan2(df['dy'], df['dx'])

    df.dropna(inplace=True)
    return df

#tworzenie etykiety danych
def add_trend(df):
    df['trend'] = np.where(df['angle'] > 0.01, 1, np.where(df['angle'] < -0.01, -1, 0))
    return  df


def read_csv_file(fileName):
    #wczytywanie danych z plików csv
    data = pd.read_csv(fileName)
    data = data.rename(
        columns={'Data': 'date', 'Otwarcie': 'open', 'Najwyzszy': 'high', 'Najnizszy': 'low', 'Zamkniecie': 'close',
                 'Wolumen': 'volume'})
    # czyszcenie danych
    data.dropna(inplace=True)
    return data


def read_all_csv_files(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    dataframes = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = read_csv_file(file_path)
        print(len(df))
        #df = df.tail(1000)
        df = add_trend_label(df)
        dataframes.append(df)
    concatenated_df = pd.concat(dataframes)
    concatenated_df.reset_index(drop=True, inplace=True)
    return concatenated_df

#0 wczytywanie danych z plików .csv
data = read_all_csv_files('files')
print(len(data))

#ustalenie długości SMA
SMA_length = 20

#1 Dodanie etykiet do danych giełdowych feature engineering
#data = add_trend_label(data, SMA_length)
stock_data_with_trend = add_trend(data)

pd.set_option('display.max_columns', None)
print(stock_data_with_trend.head())

#2 Trenowanie modelu
trained_model = train_model(stock_data_with_trend)

#3 Przewidywanie trendu dla wybranej spółki
fileName = 'files/dnp_d.csv'
data_predict = read_csv_file(fileName)

data_predict = add_trend_label(data_predict, SMA_length)
recent_data = data_predict.tail(SMA_length)

predicted_trend = predict_trend(trained_model, recent_data)
print("Przewidywany trend:", predicted_trend)

#4 agregiacja trendu
overall_trend = aggregate_trend(predicted_trend)
print("Całkowity trend w ostatnich " + str(SMA_length) + " dniach:", fileName, overall_trend)




