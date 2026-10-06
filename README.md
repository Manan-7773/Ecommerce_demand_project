# E-Commerce Product Demand Analysis — PS-04

## Features
- Best-selling and slow-moving product identification
- Increasing/decreasing/stable demand trend analysis
- Interactive demand charts
- Simple future demand forecasting using Linear Regression
- Demand-change alerts using recent-vs-previous period comparison
- Inventory recommendation based on average daily demand
- Streamlit dashboard with category and date filters

## Run
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
ecommerce_demand_project/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── orders.csv
└── assets/
```

## Dataset
The included dataset is synthetic demo order data for 12 products across 2025. Replace `data/orders.csv` with real order data using the same columns:
`date, product_id, product_name, category, unit_price, units_sold`.
