**Yes Bank Stock Closing Price Prediction**


<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/11df143a-68f9-458b-acd9-14c454b45701" />




This project builds a Time-Series Forecasting App that predicts the monthly closing price of Yes Bank Ltd. (NSE: YESBANK) using the ARIMA statistical model.
It includes a deployed-friendly Streamlit Web Application with data visualization and forecast export features.

🚀 **Project Overview:**

✔ Historical monthly stock data collected and cleaned

✔ ARIMA model trained for forecasting

✔ Streamlit dashboard for predictions

✔ Export forecasts to CSV

✔ Easy to integrate and extend for any stock or time series data

The app forecasts future monthly closing prices and visualizes both historical and predicted values.

 A great end-to-end Data Science + Time-Series + Deployment project!

📌 **Key Features:**
Feature	Description
| Feature              | Description                                 |
| -------------------- | ------------------------------------------- |
|  ARIMA Model       | Accurate univariate time-series forecasting |
|  Flexible Horizon  | Forecast next **12 / 24 / 36 months**       |
|  Streamlit App    | UI for visualization & interaction          |
|  Interactive Graph | Stock closing trend + mean forecast         |
|  Download Button   | Export forecast as CSV                      |
|  Model Saving      | Reload trained model without retraining     |


🧱 **Tech Stack:**
| Component     | Technology        |
| ------------- | ----------------- |
| Programming   | Python            |
| Data Handling | Pandas, NumPy     |
| Model         | Statsmodels ARIMA |
| Visualization | Matplotlib        |
| Deployment UI | Streamlit         |
| Serialization | Joblib            |


📂 **Folder Structure:**
yesbank-forecast/
│
├── app.py                # Streamlit Application
├── arima_model.pkl       # Trained ARIMA Model
├── data.csv              # Final cleaned Stock Data (Monthly)
├── README.md             # Documentation (This File)
└── requirements.txt      # Dependencies

▶️ **How to Run the App:**
Install dependencies:

pip install -r requirements.txt

Run Streamlit app:

streamlit run app.py


Then open browser:
 http://localhost:8501/

📁 **Dataset Used:**

Source: NSE Yes Bank historical data (Monthly)

Columns used:

Date (Month Start)

Close (Adjusted Closing Price)

Stored in project as data.csv
<img width="1163" height="449" alt="image" src="https://github.com/user-attachments/assets/f84b2bae-564b-4146-8e20-2230d550be3f" />


📈 **Model Training Summary:**

Model Type: ARIMA(p,d,q)

Target variable: Monthly Closing Price

Data Period: Up to Nov 2020

Model serialized using joblib

The model forecasts future values based solely on patterns from historical prices.

🔮 **Example Forecast Output (Next 12 Months):**
| Date       | Predicted Close |
| ---------- | --------------- |
| 2020-12-01 | 13.72           |
| 2021-01-01 | 14.49           |
| 2021-02-01 | 13.86           |
| ...        | ...             |

(Actual output may vary depending on retraining)

📦 **requirements.txt**

streamlit
pandas
numpy
matplotlib
statsmodels
joblib


🚧 **Future Enhancements**

- Confidence Interval bands in the plot
- MAPE / RMSE model accuracy displayed
- Auto model selection (ARIMA/AutoARIMA comparison)
- Add option for new data upload
- Deployment on Streamlit Cloud / Heroku

👨‍💻 **Author**

Kanika Singh Rajpoot
M.Tech — Signal Processing | Data Science Enthusiast

