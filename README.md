**Yes Bank Stock Closing Price Prediction**


<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/11df143a-68f9-458b-acd9-14c454b45701" />




This project builds a complete **time-series forecasting system** for predicting the monthly closing price of **Yes Bank Ltd. (NSE: YESBANK)**.

It includes:

- ARIMA statistical model  
- Machine-Learning model (Random Forest)  
- Hybrid model = ARIMA Trend + ML Residual Correction  
- Fully interactive Streamlit dashboard  
- Forecasts for 12 / 24 / 36 months  
- Feature engineering, visualizations, metrics & exports  

A full end-to-end Data Science + Time-Series + Deployment project.

---

## Project Highlights

- Cleaned & prepared **185+ months** of stock data  
- Built ARIMA(1,1,1) for trend modeling  
- Engineered ML features (lags, rolling mean, month, year, range)  
- Trained Random Forest for:
  - trend prediction  
  - residual correction (hybrid model)  
- Developed advanced Streamlit app with:
  - ARIMA / ML / Hybrid comparison  
  - Metrics dashboard  
  - Residual analysis  
  - Feature importance  
  - Data explorer + CSV export  
  - Multi-horizon forecasting  

---

## Models Included

### **1. ARIMA Model**
Univariate forecasting capturing:
- Trend  
- Autocorrelation  
- Month-to-month dynamics  

**Best for:** Long-term smooth trend prediction.

---

### **2. ML Model (Random Forest Regressor)**
Uses engineered features:

- Lag1, Lag2  
- 3-month Moving Average  
- Month & Year  
- High–Low price range  

**Best for:** Learning non-linear short-term movements.

---

### **3. Hybrid Model (Recommended Output)**  
**Hybrid = ARIMA Mean + ML Residual Prediction**

Benefits:
- ARIMA handles trend  
- ML corrects volatility  
- Best accuracy overall  

---

## Example Forecast (12 Months)

| Date       | ARIMA | ML Trend | Hybrid (Final) |
|------------|-------|----------|----------------|
| 2020-12-01 | 13.72 | 14.77    | 14.21          |
| 2021-01-01 | 14.49 | 14.45    | 14.74          |
| 2021-02-01 | 13.86 | 13.51    | 14.12          |
| ...        | ...   | ...      | ...            |

(The hybrid model gives the most realistic future curve.)

---

## 📁 Folder Structure
yesbank-forecast/
│── app.py # Streamlit Application
│── data.csv # Cleaned Monthly Stock Data
│── arima_model.pkl # Saved ARIMA model
│── best_ml_model.pkl # ML Trend Model
│── rf_resid.pkl # Residual Correction Model
│── scaler.pkl # Scaler for ML
│── scaler_resid.pkl # Scaler for residual model
│── features.json # Saved ML feature list
│── requirements.txt
│── README.md


---

## Tech Stack

| Component     | Technology |
|---------------|------------|
| Programming   | Python     |
| Time-Series   | Statsmodels ARIMA |
| ML Models     | Scikit-Learn |
| Deployment    | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Serialization | Joblib |
| Notebook      | Google Colab |

---

## How to Run

Install dependencies:

pip install -r requirements.txt

Run Streamlit app:

streamlit run app.py



Open browser →  
**http://localhost:8501**

---

## Accuracy Metrics

- ARIMA RMSE: ~46  
- RandomForest RMSE: ~7.6  
- Improvement: **84% reduction in RMSE**  
- Hybrid improves short-term movement prediction  
- ARIMA provides stable backbone  
- ML corrects volatility (residual learning)

---

## Visualizations Included

- Historical Stock Trend  
- ARIMA vs ML vs Hybrid Comparison  
- Hybrid Decomposition (ARIMA trend + residual correction)  
- Feature Importance  
- Rolling Statistics  
- Residual Plots  
- Data Explorer with filtering and export  

---

 **Dataset Used:**

Source: NSE Yes Bank historical data (Monthly)

Columns used:

Date (Month Start)

Close (Adjusted Closing Price)

Stored in project as data.csv
<img width="1163" height="449" alt="image" src="https://github.com/user-attachments/assets/f84b2bae-564b-4146-8e20-2230d550be3f" />

---

 **requirements.txt**

streamlit
pandas
numpy
matplotlib
statsmodels
joblib

---

## Future Enhancements

- LSTM / Prophet model comparisons  
- Auto-ARIMA  
- Include macro-economic indicators  
- API-based real-time stock updates  
- Deployment on Streamlit Cloud  

---

 **Author**

Kanika Singh Rajpoot
M.Tech — Signal Processing | Data Science Enthusiast

