# [cite_start]Mô hình hóa và Dự báo nhu cầu điện (Electricity Load Diagrams) [cite: 1, 4]

## 1. Giới thiệu
[cite_start]Dự án tập trung vào phân tích tính thời vụ phức tạp và dự báo nhu cầu điện năng[cite: 4, 95].

## 2. Dữ liệu
[cite_start]Sử dụng tập dữ liệu "Electricity Load Diagrams" từ HuggingFace/GluonTS[cite: 6, 7].
- [cite_start]Tần suất: Hàng giờ/15 phút[cite: 16].
- [cite_start]Quy mô: Nhiều chuỗi thời gian (người tiêu dùng)[cite: 15].

## [cite_start]3. Các mô hình triển khai [cite: 47]
- [cite_start]Baseline: ARIMA, SARIMA[cite: 49, 106].
- [cite_start]Machine Learning: XGBoost, LightGBM[cite: 55].
- [cite_start]Deep Learning: LSTM, Transformer (thông qua GluonTS)[cite: 58, 59].

## [cite_start]4. Đánh giá [cite: 60]
[cite_start]Sử dụng các chỉ số: MAE, RMSE, MAPE[cite: 63, 137].