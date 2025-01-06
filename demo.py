import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from matplotlib.dates import YearLocator, DateFormatter, MonthLocator  # Import bổ sung

# Hàm lấy dữ liệu lịch sử cổ phiếu
def get_stock_history(symbol, start_date, end_date):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            print("Không tìm thấy dữ liệu cho mã chứng khoán này.")
            return None
        df = df.drop(['Dividends', 'Stock Splits'], axis=1)
        df = df.reset_index()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
        return df
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return None

# Phần chính của chương trình
if __name__ == "__main__":
    symbol = input("Nhập mã chứng khoán (ACB.VN, TCB.VN, etc): ").strip().upper()
    print("Chọn khoảng thời gian:")
    print("1. Số ngày trước (nhập: 'd')")
    print("2. Số tháng trước (nhập: 'm')")
    print("3. Số năm trước (nhập: 'y')")
    unit = input("Nhập đơn vị thời gian (d/m/y): ").strip().lower()
    quantity = int(input("Nhập số lượng (ví dụ: 30 cho 30 ngày): ").strip())
    today = datetime.now()
    if unit == 'd':
        start_date = today - timedelta(days=quantity)
    elif unit == 'm':
        start_date = today - timedelta(days=quantity * 30)
    elif unit == 'y':
        start_date = today - timedelta(days=quantity * 365)
    else:
        print("Đơn vị thời gian không hợp lệ!")
        exit()
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    df = get_stock_history(symbol, start_date, end_date)
    output_folder = "stock_data"
    if df is not None:
        print("\nData retrieved:")
        print(df.head())
        output_path = os.path.join(output_folder, f"{symbol}_yahoo_history.csv")
        df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")

        # Tiến hành phân tích và dự đoán từ dữ liệu
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        df = df.fillna(df.mean())
        df = df.sort_values(by='Date')

        plt.figure(figsize=(10, 5))
        plt.plot(df['Date'], df['Close'], label='Giá đóng cửa', color='red')
        plt.xlabel('Năm')
        plt.ylabel('Giá đóng cửa')
        plt.title('Biểu đồ giá đóng cửa của cổ phiếu qua các năm')
        plt.legend(loc='best')
        years = YearLocator()
        yearsFmt = DateFormatter('%Y')
        months = MonthLocator()
        plt.gca().xaxis.set_major_locator(years)
        plt.gca().xaxis.set_major_formatter(yearsFmt)
        plt.gca().xaxis.set_minor_locator(months)
        plt.tight_layout()
        plt.show()

        df1 = pd.DataFrame(df, columns=['Date', 'Close'])
        df1.index = df1.Date
        df1.drop('Date', axis=1, inplace=True)
        data = df1.values
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]

        sc = MinMaxScaler(feature_range=(0, 1))
        sc_train = sc.fit_transform(data)

        x_train, y_train = [], []
        for i in range(50, len(train_data)):
            x_train.append(sc_train[i-50:i, 0])
            y_train.append(sc_train[i, 0])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(Input(shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=128, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.compile(loss='mean_absolute_error', optimizer='adam')

        save_model = "save_model.keras"
        best_model = ModelCheckpoint(save_model, monitor='loss', verbose=2, save_best_only=True, mode='auto')
        model.fit(x_train, y_train, epochs=110, batch_size=128, verbose=2, callbacks=[best_model])

        y_train = sc.inverse_transform(y_train.reshape(-1, 1))
        final_model = load_model("save_model.keras")
        y_train_predict = final_model.predict(x_train)
        y_train_predict = sc.inverse_transform(y_train_predict)

        test = df1[len(train_data)-50:].values
        test = test.reshape(-1, 1)
        sc_test = sc.transform(test)

        x_test = []
        for i in range(50, test.shape[0]):
            x_test.append(sc_test[i-50:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        y_test = data[train_size:]
        y_test_predict = final_model.predict(x_test)
        y_test_predict = sc.inverse_transform(y_test_predict)

        train_data1 = df1[50:train_size].copy()
        test_data1 = df1[train_size:].copy()
        train_data1.loc[:, 'Dự đoán'] = y_train_predict
        test_data1.loc[:, 'Dự đoán'] = y_test_predict

        plt.figure(figsize=(24, 8))
        plt.plot(df1, label='Giá thực tế', color='red')
        plt.plot(train_data1['Dự đoán'], label='Giá dự đoán train', color='green')
        plt.plot(test_data1['Dự đoán'], label='Giá dự đoán test', color='blue')
        plt.title('So sánh giá dự báo và giá thực tế')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá đóng cửa (VNĐ)')
        plt.legend()
        plt.show()

        print('Độ phù hợp tập train:', r2_score(y_train, y_train_predict))
        print('Sai số tuyệt đối trung bình trên tập train (VNĐ):', mean_absolute_error(y_train, y_train_predict))
        print('Phần trăm sai số tuyệt đối trung bình tập train:', mean_absolute_percentage_error(y_train, y_train_predict))
        print('Độ phù hợp tập test:', r2_score(y_test, y_test_predict))
        print('Sai số tuyệt đối trung bình trên tập test (VNĐ):', mean_absolute_error(y_test, y_test_predict))
        print('Phần trăm sai số tuyệt đối trung bình tập test:', mean_absolute_percentage_error(y_test, y_test_predict))

        next_date = df['Date'].iloc[-1] + pd.Timedelta(days=1)
        x_next = np.array([sc_train[-50:, 0]])
        x_next = np.reshape(x_next, (x_next.shape[0], x_next.shape[1], 1))
        y_next_predict = final_model.predict(x_next)
        y_next_predict = sc.inverse_transform(y_next_predict)

        df_next = pd.DataFrame({'Date': [next_date], 'Close': [y_next_predict[0][0]]})
        df1 = pd.concat([df1, df_next])

        plt.figure(figsize=(15, 5))
        plt.plot(df1.index, df1['Close'], label='Giá thực tế', color='red')
        plt.plot(train_data1.index, train_data1['Dự đoán'], label='Giá dự đoán train', color='green')
        plt.plot(test_data1.index, test_data1['Dự đoán'], label='Giá dự đoán test', color='blue')
        plt.scatter([next_date], [y_next_predict[0][0]], color='orange', label='Dự đoán ngày kế tiếp')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá đóng cửa (VNĐ)')
        plt.title('So sánh giá dự báo và giá thực tế')
        plt.legend()
        plt.show()

        actual_closing_price = df['Close'].iloc[-1]
        comparison_df = pd.DataFrame({'Date': [next_date], 'Giá dự đoán': [y_next_predict[0][0]], 'Giá ngày trước': [actual_closing_price]})
        print(comparison_df)