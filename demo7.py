import pandas as pd  # Đọc dữ liệu
import numpy as np  # Xử lý dữ liệu
import matplotlib.pyplot as plt  # Vẽ biểu đồ
from sklearn.preprocessing import MinMaxScaler  # Chuẩn hóa dữ liệu
from keras.callbacks import ModelCheckpoint  # Lưu lại huấn luyện tốt nhất
from tensorflow.keras.models import load_model  # Tải mô hình

# Các lớp để xây dựng mô hình
from keras.models import Sequential  # Đầu vào
from keras.layers import LSTM, Dropout, Dense  # Học phụ thuộc, tránh học tủ, đầu ra
from keras.layers import Input  # Định nghĩa đầu vào

# Kiểm tra độ chính xác của mô hình
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error  # Đo mức độ phù hợp, sai số tuyệt đối trung bình, % sai số tuyệt đối trung bình

# Đọc dữ liệu từ file CSV
df = pd.read_csv('ACB.VN_yahoo_history.csv')

# Hiển thị lại DataFrame sau khi xóa
print(df)

# Định dạng cấu trúc thời gian
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df = df.fillna(df.mean())
# Kích thước dữ liệu
print(df.shape)

# Dữ liệu 5 dòng đầu
print(df.head())

# Mô tả bộ dữ liệu
print(df.describe())

from matplotlib.dates import YearLocator, DateFormatter, MonthLocator

# Chuyển đổi cột "Date" sang dạng datetime
df = df.sort_values(by='Date')

# Tạo đồ thị giá đóng cửa qua các năm
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Close'], label='Giá đóng cửa', color='red')
plt.xlabel('Năm')
plt.ylabel('Giá đóng cửa')
plt.title('Biểu đồ giá đóng cửa của ACB qua các năm')
plt.legend(loc='best')

# Định dạng đồ thị hiển thị các ngày tháng theo năm-tháng
years = YearLocator()
yearsFmt = DateFormatter('%Y')
months = MonthLocator()  # Thêm dòng này để khai báo MonthLocator
plt.gca().xaxis.set_major_locator(years)
plt.gca().xaxis.set_major_formatter(yearsFmt)
plt.gca().xaxis.set_minor_locator(months)

plt.tight_layout()
plt.show()

# Chuẩn bị dữ liệu để huấn luyện mô hình
df1 = pd.DataFrame(df, columns=['Date', 'Close'])
df1.index = df1.Date
df1.drop('Date', axis=1, inplace=True)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
data = df1.values
train_data = data[:1500]
test_data = data[1500:]

# Chuẩn hóa dữ liệu
sc = MinMaxScaler(feature_range=(0, 1))
sc_train = sc.fit_transform(data)

# Tạo vòng lặp các giá trị
x_train, y_train = [], []
for i in range(50, len(train_data)):
    x_train.append(sc_train[i-50:i, 0])  # Lấy 50 giá đóng cửa liên tục
    y_train.append(sc_train[i, 0])  # Lấy ra giá đóng cửa ngày hôm sau

# Xếp dữ liệu thành 1 mảng 2 chiều
x_train = np.array(x_train)
y_train = np.array(y_train)

# Xếp lại dữ liệu thành mảng 1 chiều
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Xây dựng mô hình
model = Sequential()  # Tạo lớp mạng cho dữ liệu đầu vào

# Định nghĩa lớp đầu vào
model.add(Input(shape=(x_train.shape[1], 1)))

# 2 lớp LSTM
model.add(LSTM(units=128, return_sequences=False))  # Học phụ thuộc dài hạn
# model.add(LSTM(units=64))
model.add(Dropout(0.5))  # Loại bỏ 1 số đơn vị tránh học tủ (overfitting)
model.add(Dense(1))  # Output đầu ra 1 chiều

# Đo sai số tuyệt đối trung bình có sử dụng trình tối ưu hóa adam
model.compile(loss='mean_absolute_error', optimizer='adam')

# Huấn luyện mô hình
save_model = "save_model.keras"
best_model = ModelCheckpoint(save_model, monitor='loss', verbose=2, save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=110, batch_size=128, verbose=2, callbacks=[best_model])

# Dữ liệu train
y_train = sc.inverse_transform(y_train.reshape(-1, 1))  # Giá thực
final_model = load_model("save_model.keras")
y_train_predict = final_model.predict(x_train)  # Dự đoán giá đóng cửa trên tập đã train
y_train_predict = sc.inverse_transform(y_train_predict)  # Giá dự đoán

# Xử lý dữ liệu test
test = df1[len(train_data)-50:].values
test = test.reshape(-1, 1)
sc_test = sc.transform(test)

x_test = []
for i in range(50, test.shape[0]):
    x_test.append(sc_test[i-50:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Dữ liệu test
y_test = data[1500:]  # Giá thực
y_test_predict = final_model.predict(x_test)
y_test_predict = sc.inverse_transform(y_test_predict)  # Giá dự đoán

# Lập biểu đồ so sánh
train_data1 = df1[50:1500].copy()
test_data1 = df1[1500:].copy()

train_data1.loc[:, 'Dự đoán'] = y_train_predict  # Thêm dữ liệu
test_data1.loc[:, 'Dự đoán'] = y_test_predict  # Thêm dữ liệu

plt.figure(figsize=(24, 8))
plt.plot(df1, label='Giá thực tế', color='red')  # Đường giá thực
plt.plot(train_data1['Dự đoán'], label='Giá dự đoán train', color='green')  # Đường giá dự báo train
plt.plot(test_data1['Dự đoán'], label='Giá dự đoán test', color='blue')  # Đường giá dự báo test
plt.title('So sánh giá dự báo và giá thực tế')  # Đặt tên biểu đồ
plt.xlabel('Thời gian')  # Đặt tên hàm x
plt.ylabel('Giá đóng cửa (VNĐ)')  # Đặt tên hàm y
plt.legend()  # Chú thích
plt.show()

# Đánh giá mô hình trên tập train
print('Độ phù hợp tập train:', r2_score(y_train, y_train_predict))
print('Sai số tuyệt đối trung bình trên tập train (VNĐ):', mean_absolute_error(y_train, y_train_predict))
print('Phần trăm sai số tuyệt đối trung bình tập train:', mean_absolute_percentage_error(y_train, y_train_predict))

# Đánh giá mô hình trên tập test
print('Độ phù hợp tập test:', r2_score(y_test, y_test_predict))
print('Sai số tuyệt đối trung bình trên tập test (VNĐ):', mean_absolute_error(y_test, y_test_predict))
print('Phần trăm sai số tuyệt đối trung bình tập test:', mean_absolute_percentage_error(y_test, y_test_predict))

# Lấy ngày kế tiếp sau ngày cuối cùng trong tập dữ liệu để dự đoán
next_date = df['Date'].iloc[-1] + pd.Timedelta(days=1)

# Chuẩn hóa giá trị của ngày cuối cùng
x_next = np.array([sc_train[-50:, 0]])  # Lấy 50 giá đóng cửa gần nhất
x_next = np.reshape(x_next, (x_next.shape[0], x_next.shape[1], 1))
y_next_predict = final_model.predict(x_next)
y_next_predict = sc.inverse_transform(y_next_predict)

# Thêm dữ liệu dự đoán của ngày kế tiếp vào DataFrame
df_next = pd.DataFrame({'Date': [next_date], 'Close': [y_next_predict[0][0]]})
df1 = pd.concat([df1, df_next])

# Vẽ biểu đồ mới với dự đoán cho ngày kế tiếp
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

# Lấy giá trị của ngày cuối cùng trong tập dữ liệu
actual_closing_price = df['Close'].iloc[-1]

# Tạo DataFrame so sánh giá dự đoán với giá ngày cuối trong tập dữ liệu
comparison_df = pd.DataFrame({'Date': [next_date], 'Giá dự đoán': [y_next_predict[0][0]], 'Giá ngày trước': [actual_closing_price]})

# In ra bảng so sánh
print(comparison_df)