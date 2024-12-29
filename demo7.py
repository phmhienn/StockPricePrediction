import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.models import Model

# Đọc dữ liệu từ file csv local
try:
    # Đọc cả dữ liệu train và test
    train_df = pd.read_csv('ACB.VN_train.csv')
    test_df = pd.read_csv('ACB.VN_test.csv')
    print("Đã đọc dữ liệu thành công!")
    
    # Gộp dữ liệu
    df = pd.concat([train_df, test_df])
except FileNotFoundError:
    print("Không tìm thấy file dữ liệu! Vui lòng kiểm tra lại đường dẫn.")
    exit()

# Định dạng cấu trúc thời gian
df["Date"] = pd.to_datetime(df.Date)
df = df.sort_values(by='Date')  # Sắp xếp theo thời gian

# Tạo DataFrame cho dữ liệu huấn luyện
df1 = pd.DataFrame(df, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
df1.index = df1.Date
df1.drop('Date', axis=1, inplace=True)

# Chia tập dữ liệu
data = df1.values
train_size = int(len(train_df))
train_data = data[:train_size]
test_data = data[train_size:]

# Chuẩn hóa tất cả các features
sc = MinMaxScaler(feature_range=(0,1))
scaled_data = sc.fit_transform(df1)

# Tạo chuỗi dữ liệu cho huấn luyện
lookback = 50
x_train, y_train = [], []
for i in range(lookback, len(train_data)):
    x_train.append(scaled_data[i-lookback:i, 0])
    y_train.append(scaled_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1))

# Xây dựng mô hình
def build_model(lookback):
    inputs = Input(shape=(lookback, 1))
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='huber',
        metrics=['mae', 'mse']
    )
    return model

# Tạo mô hình
model = build_model(lookback)

# Lưu mô hình tốt nhất
save_model = "best_model_acb.keras"
best_model = ModelCheckpoint(
    save_model, 
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='auto'
)

# Thêm early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Huấn luyện mô hình
history = model.fit(
    x_train, 
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, best_model],
    verbose=1
)

# Dự đoán
final_model = load_model("best_model_acb.keras")

def predict_future_prices(model, last_sequence, sc, n_days=30):
    """
    Dự đoán giá với kiểm soát biến động
    """
    future_prices = []
    current_sequence = last_sequence[-lookback:].copy()
    last_price = current_sequence[-1]  # Giá của ngày cuối cùng
    
    for _ in range(n_days):
        # Chuẩn bị dữ liệu
        current_sequence_scaled = current_sequence.reshape(-1, 1)
        current_sequence_scaled = np.repeat(current_sequence_scaled, 5, axis=1)
        current_sequence_scaled = sc.transform(current_sequence_scaled)[:, 0]
        X = current_sequence_scaled.reshape(1, lookback, 1)
        
        # Dự đoán
        next_day = model.predict(X, verbose=0)
        next_day_array = np.array([[next_day[0, 0]]*5])
        next_day_price = sc.inverse_transform(next_day_array)[0, 0]
        
        # Giới hạn biến động giá (tối đa 2% mỗi ngày)
        max_daily_change = 0.02  # 2%
        min_price = last_price * (1 - max_daily_change)
        max_price = last_price * (1 + max_daily_change)
        next_day_price = np.clip(next_day_price, min_price, max_price)
        
        # Thêm kiểm tra xu hướng
        if len(future_prices) >= 5:
            # Kiểm tra xu hướng 5 ngày gần nhất
            recent_trend = (next_day_price - future_prices[-5]) / future_prices[-5]
            if abs(recent_trend) > 0.1:  # Nếu xu hướng > 10%
                # Điều chỉnh giá về mức hợp lý hơn
                next_day_price = future_prices[-1] * (1 + np.sign(recent_trend) * 0.01)
        
        future_prices.append(next_day_price)
        current_sequence = np.append(current_sequence[1:], next_day_price)
        last_price = next_day_price
    
    return future_prices

# Lấy chuỗi dữ liệu cuối cùng để dự đoán
last_sequence = df1['Close'].values[-lookback:]  # Chỉ lấy lookback ngày cuối

# Dự đoán 30 ngày tiếp theo
future_days = 30
future_prices = predict_future_prices(final_model, last_sequence, sc, future_days)

# Tạo ngày cho dự đoán
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), 
    periods=future_days, 
    freq='B'
)

# Tạo DataFrame kết quả
results = pd.DataFrame({
    'Ngày': future_dates,
    'Giá dự đoán': future_prices
})

# Vẽ biểu đồ
plt.figure(figsize=(15, 7))
plt.plot(df1.index[-30:], df1['Close'].values[-30:], 'b-', label='Giá lịch sử')
plt.plot(future_dates, future_prices, 'r--', label='Giá dự đoán')
plt.title('Dự đoán giá cổ phiếu ACB')
plt.xlabel('Thời gian')
plt.ylabel('Giá (VNĐ)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# In kết quả dự đoán
print("\nDự đoán giá cho 30 ngày tới:")
print(results.to_string(index=False))

# Lưu kết quả
results.to_csv('ket_qua_du_doan_ACB.csv', index=False, encoding='utf-8-sig')
print("\nĐã lưu kết quả vào file 'ket_qua_du_doan_ACB.csv'")

def calculate_confidence_interval(predictions, confidence=0.95):
    """
    Tính toán khoảng tin cậy cho dự đoán
    """
    mean = np.mean(predictions)
    std = np.std(predictions)
    z_score = 1.96  # cho confidence level 95%
    margin = z_score * (std / np.sqrt(len(predictions)))
    return mean - margin, mean + margin

# Thêm hàm đánh giá độ tin cậy
def add_confidence_intervals(results_df, confidence=0.95):
    """
    Thêm khoảng tin cậy cho dự đoán
    """
    z_score = 1.96  # 95% confidence interval
    std = results_df['Giá dự đoán'].std()
    margin = z_score * (std / np.sqrt(len(results_df)))
    
    results_df['Khoảng dưới'] = results_df['Giá dự đoán'] - margin
    results_df['Khoảng trên'] = results_df['Giá dự đoán'] + margin
    return results_df


