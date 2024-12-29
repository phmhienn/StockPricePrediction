import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_history(symbol, start_date, end_date):
    try:
        stock = yf.Ticker(symbol)

        # Lấy dữ liệu với start và end date
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            print("Không tìm thấy dữ liệu cho mã chứng khoán này.")
            return None

        # Remove Dividends and Stock Splits columns
        df = df.drop(['Dividends', 'Stock Splits'], axis=1)
        df = df.reset_index()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
        return df
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return None

if __name__ == "__main__":
    symbol = input("Nhập mã chứng khoán (ACB.VN, VCB.VN, etc): ").strip().upper()
    print("Chọn khoảng thời gian:")
    print("1. Số ngày trước (nhập: 'd')")
    print("2. Số tháng trước (nhập: 'm')")
    print("3. Số năm trước (nhập: 'y')")
    unit = input("Nhập đơn vị thời gian (d/m/y): ").strip().lower()
    quantity = int(input("Nhập số lượng (ví dụ: 30 cho 30 ngày): ").strip())

    # Tính toán ngày bắt đầu và ngày kết thúc
    today = datetime.now()
    if unit == 'd':
        start_date = today - timedelta(days=quantity)
    elif unit == 'm':
        start_date = today - timedelta(days=quantity * 30)  # 1 tháng = 30 ngày
    elif unit == 'y':
        start_date = today - timedelta(days=quantity * 365)  # 1 năm = 365 ngày
    else:
        print("Đơn vị thời gian không hợp lệ!")
        exit()
    
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    # Lấy dữ liệu
    df = get_stock_history(symbol, start_date, end_date)
    
    if df is not None:
        print("\nData retrieved:")
        print(df.head())
        df.to_csv(f"{symbol}_yahoo_history.csv", index=False)
        print(f"\nSaved to {symbol}_yahoo_history.csv")
