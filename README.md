# Stock Future Visualizer

A Streamlit app for visualizing possible future stock price ranges from company fundamentals, valuation assumptions, and scenario modeling.

## Features

- Search by stock ticker.
- Fetch current company data from Yahoo Finance.
- Model revenue growth, net margin changes, and share count changes.
- Compare bear, base, and bull scenarios in one chart.
- Build custom scenarios with your own growth, margin, share count, and P/E assumptions.
- Normalize unusually high growth rates over time.
- View annual operating assumptions, projected EPS, projected price ranges, analyst targets, valuation context, and historical price performance.
- Review data quality notes when Yahoo Finance fields are missing or defaulted.

## Tech Stack

- Python
- Streamlit
- yfinance
- pandas
- Altair

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer

This app is for educational and exploratory use only. It is not financial advice.
