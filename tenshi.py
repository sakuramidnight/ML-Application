from PIL import Image
import numpy as np
from Tenshi_RBCP import str_to_date
from Tenshi_RBCP import df_to_windowed_df
from Tenshi_RBCP import windowed_df_to_date_x_y
import keras
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

model_1 = keras.models.load_model('mahiru.keras')
model_2 = keras.models.load_model('mahiru_alt.keras')
model_3 = keras.models.load_model('mahiru_rbcp2.keras')
model_4 = keras.models.load_model('mahiru_rbcp3.keras')

image = Image.open('suisei.png')


def plot_prediction_single(date, pred, string):
    plt.figure(figsize=(10, 8))
    plt.title('Prediction')
    plt.plot(date, pred)
    plt.xlabel('Time')
    plt.ylabel('Weight (kg)')
    plt.legend([f"{string}"])
    st.pyplot(plt.gcf())



def plot_prediction_all(d1, d2, d3, d4, p1, p2, p3, p4):
    plt.figure(figsize=(10, 8))
    plt.plot(d1, p1)
    plt.plot(d2, p2)
    plt.plot(d3, p3)
    plt.plot(d4, p4)
    plt.xlabel('Time')
    plt.ylabel('Weight (kg)')
    plt.legend(['Food Item 1', 'Food Item 2', 'Food Item 3', 'Food Item 4'])
    st.pyplot(plt.gcf())



def main():
    st.set_page_config(
        page_title='Exigency',
        layout='centered',
        page_icon=image
    )

    st.markdown("""
                <style>
                    div[data-testid="column"] {
                        width: fit-content !important;
                        flex: unset;
                    }
                    div[data-testid="column"] * {
                        width: fit-content !important;
                    }
                </style>
                """, unsafe_allow_html=True)

    st.markdown("", unsafe_allow_html=True)

    st.header('Food Waste Reduction: Predicting the amount of food to be procured')
    st.text('Please upload weight data to begin the prediction process')

    uploaded_file = st.file_uploader("Upload your file", type=["csv"], key="uploaded_file")

    if uploaded_file is not None:
        # Set up date in dataframe
        dataframe = pd.read_csv(uploaded_file)

        # View data
        st.table(dataframe[:10])

        if st.button('Click to view the full data'):
            st.table(dataframe)

        # Select data for prediction/ Transform data
        select_data = st.selectbox('Select data for prediction', ['Food Item 1', 'Food Item 2', 'Food Item 3',
                                                                  'Food Item 4', 'All Data'])


        # Select prediction range
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            start_data = st.text_input('Select the first date (MM/DD/YY)')
        with c2:
            end_data = st.text_input('Select the last date (MM/DD/YY)')
        with c3:
            range_alt = st.text_input('Select the range')

        # Confirmation/ Prediction
        if st.button('Confirm your choices'):
            st.text('Your choice has been confirmed, proceed to the prediction stage...')
            if select_data == 'All Data':
                data_1 = dataframe[['Date', 'Food Item 1']]
                data_1['Date'] = data_1['Date'].apply(str_to_date)
                data_1.index = data_1.pop('Date')

                data_2 = dataframe[['Date', 'Food Item 2']]
                data_2['Date'] = data_2['Date'].apply(str_to_date)
                data_2.index = data_2.pop('Date')

                data_3 = dataframe[['Date', 'Food Item 3']]
                data_3['Date'] = data_3['Date'].apply(str_to_date)
                data_3.index = data_3.pop('Date')

                data_4 = dataframe[['Date', 'Food Item 4']]
                data_4['Date'] = data_4['Date'].apply(str_to_date)
                data_4.index = data_4.pop('Date')

                LSTM_df_1 = df_to_windowed_df(data_1, str(start_data), str(end_data),
                                              int(range_alt), string='Food Item 1')
                LSTM_df_2 = df_to_windowed_df(data_2, str(start_data), str(end_data),
                                              int(range_alt), string='Food Item 2')
                LSTM_df_3 = df_to_windowed_df(data_3, str(start_data), str(end_data),
                                              int(range_alt), string='Food Item 3')
                LSTM_df_4 = df_to_windowed_df(data_4, str(start_data), str(end_data),
                                              int(range_alt), string='Food Item 4')


                date_1, in_1, out_1 = windowed_df_to_date_x_y(LSTM_df_1)
                date_2, in_2, out_2 = windowed_df_to_date_x_y(LSTM_df_2)
                date_3, in_3, out_3 = windowed_df_to_date_x_y(LSTM_df_3)
                date_4, in_4, out_4 = windowed_df_to_date_x_y(LSTM_df_4)

                res_1 = model_1.predict(in_1)
                res_2 = model_2.predict(in_2)
                res_3 = model_3.predict(in_3)
                res_4 = model_4.predict(in_4)

                plot_prediction_all(date_1, date_2, date_3, date_4, res_1, res_2, res_3, res_4)
                st.text('Prediction completed, results illustrated')

                pred_data = {
                    
                }

            else:
                data_alt = dataframe[['Date', select_data]]
                data_alt['Date'] = data_alt['Date'].apply(str_to_date)
                data_alt.index = data_alt.pop('Date')

                LSTM_df = df_to_windowed_df(data_alt, str(start_data), str(end_data), int(range_alt),
                                            string=select_data)
                date, input, output = windowed_df_to_date_x_y(LSTM_df)
                if select_data == 'Food Item 1':
                    res = model_1.predict(input)
                    plot_prediction_single(date, res, select_data)
                    st.text('Prediction completed, results illustrated')
                if select_data == 'Food Item 2':
                    res = model_2.predict(input)
                    plot_prediction_single(date, res, select_data)
                    st.text('Prediction completed, results illustrated')
                if select_data == 'Food Item 3':
                    res = model_3.predict(input)
                    plot_prediction_single(date, res, select_data)
                    st.text('Prediction completed, results illustrated')
                if select_data == 'Food Item 4':
                    res = model_4.predict(input)
                    plot_prediction_single(date, res, select_data)
                    st.text('Prediction completed, results illustrated')




if __name__ == '__main__':
    main()

