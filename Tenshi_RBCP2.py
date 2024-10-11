import torch
import keras
from keras import layers
from tensorflow.keras.optimizers import Adam

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle


device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32

df = pd.read_csv('../../Pytorch/FW Data.csv')
data = df[['Date', 'Food Item 2']]


def str_to_date(date_alt):
    split = date_alt.split('/')
    y, m, d = int(split[2]), int(split[0]), int(split[1])
    return datetime.datetime(y, m, d)


data['Date'] = data['Date'].apply(str_to_date)
data.index = data.pop('Date')


def df_to_windowed_df(dataframe, first_date_str, last_date_str, n):
    first_date = str_to_date(first_date_str)
    last_date = str_to_date(last_date_str)

    target_date = first_date

    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n + 1)

        if len(df_subset) != n + 1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['Food Item 2'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
        X_alt = X[:, i]
        ret_df[f'Target-{n - i}'] = X_alt

    ret_df['Target'] = Y

    return ret_df


LSTM_df = df_to_windowed_df(dataframe=data, first_date_str='1/1/2023', last_date_str='8/29/2024', n=6)


def windowed_df_to_date_x_y(windowed_df):
    df_as_np = windowed_df.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1:-1]
    x = middle_matrix.reshape(len(dates), middle_matrix.shape[1], 1)
    y = df_as_np[:, -1]

    return dates, x.astype(np.float32), y.astype(np.float32)


date, input, output = windowed_df_to_date_x_y(LSTM_df)

train_split = int(len(date[:-8]) * 0.8)
val_split = int(len(date[:-8]) * 0.9)

train_date, train_x, train_y = date[:train_split], input[:train_split], output[:train_split]
val_date, val_x, val_y = date[train_split:val_split], input[train_split:val_split], output[train_split:val_split]
test_date, test_x, test_y = date[val_split:], input[val_split:], output[val_split:]

mahiru_alt = keras.Sequential([layers.Input((6, 1)),
                               layers.LSTM(64),
                               layers.Dense(32, activation='relu'),
                               layers.Dense(32, activation='relu'),
                               layers.Dense(1)])

mahiru_alt.compile(loss='mse', optimizer=Adam(learning_rate=0.001),
                   metrics=['mean_absolute_error'])
mahiru_alt = keras.models.load_model('../../Pytorch/mahiru_alt.keras')
model_file = 'tenshi_rbcp2.pkl'

with open(model_file, 'wb') as file:
    pickle.dump(mahiru_alt, file)

