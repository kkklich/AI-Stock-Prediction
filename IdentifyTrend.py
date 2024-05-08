import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

# Wczytaj dane historyczne cen akcji
data = pd.read_csv('pzu_d (3).csv')

data = data.rename(columns={'Data': 'Date', 'Otwarcie': 'Open','Najwyzszy':'High','Najnizszy':'Low','Zamkniecie':'Close', 'Wolumen':'Volume'})
#print(data)

# Inżynieria cech: Dodaj średnią kroczącą jako cechę
#window_size = 10
#data['rolling_mean'] = data['Close'].rolling(window=window_size).mean()
# Usuń wiersze z brakującymi wartościami
data.dropna(inplace=True)
print(data)
print('------')


def identify_trend_with_regression(stock_data):
    # Extracting the 'Close' prices from the stock data
    #close_prices = [day['Close'] for day in stock_data]


    close_prices = stock_data['Close'].tolist()

    # Creating X (indices of data points) and y (Close prices)
    X = np.arange(len(close_prices)).reshape(-1, 1)
    y = np.array(close_prices)

    # Fitting a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predicting the next data point
    next_data_point = len(X)
    next_price = model.predict([[next_data_point]])[0]

    # Determining the trend based on the predicted price compared to the last actual price
    if next_price > close_prices[-1]:
        trend = 'Predicted upward trend'
    elif next_price < close_prices[-1]:
        trend = 'Predicted downward trend'
    else:
        trend = 'No clear trend'

    print(next_data_point, next_price)
    print(trend)
     # Plotting the data
    plt.plot(X, y, label='Actual Prices')
    plt.plot(X, model.predict(X), label='Predicted Prices')
    plt.scatter(next_data_point, next_price, color='red', label='Next Predicted Price')
    plt.xlabel('Data Point')
    plt.ylabel('Price')
    plt.title('Data and Predictions')
    plt.legend()
    plt.show()

    return trend


import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def classify_stock_trend(data):
    # Assuming that the data is an array of {Date, Open, High, Low, Close, Volume}
    # and that the features are the past price history and the label is the future price trend

    # Extract 'Close' values
    #close_values = [row[4] for row in data]
    close_values = data['Close'].tolist()

    # Split the data into features (X) and labels (y)
    #X = np.array([[data[i - 1][1], data[i - 1][2], data[i - 1][3], data[i - 1][4]] for i in range(1, len(data))])
    #y = np.array(close_values[1:])

    data['rolling_mean'] = data['Close'].rolling(window=20).mean()
    data.dropna(inplace=True)

    X = data[['Close']].values
    y = data['rolling_mean'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = knn.predict(X_test)

    # Print the classification report
    print(classification_report(y_test, y_pred))


# Calling the function with the example stock data
#last_1500_rows = data.tail(1900)
last_1500_rows = data

#predicted_trend = identify_trend_with_regression(last_1500_rows)
#print(predicted_trend)

classify_stock_trend(last_1500_rows)