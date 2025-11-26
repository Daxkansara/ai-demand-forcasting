import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("Demand_Forecasting.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['store','item','date']).reset_index(drop=True)

from sklearn.preprocessing import LabelEncoder

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# ================================
# FEATURE ENGINEERING
# ================================
def add_features(data):
    d = data.copy()
    d['lag_1'] = d.groupby(['store','item'])['sales'].shift(1)
    d['lag_7'] = d.groupby(['store','item'])['sales'].shift(7)
    d['rolling_7'] = d.groupby(['store','item'])['sales'].shift(1).rolling(7).mean()
    d['rolling_14'] = d.groupby(['store','item'])['sales'].shift(1).rolling(14).mean()
    d['day'] = d['date'].dt.day
    d['month'] = d['date'].dt.month
    d['weekday'] = d['date'].dt.weekday
    return d

df = add_features(df)
df = df.dropna()

train = df[df['date'] < df['date'].max() - pd.Timedelta(days=7)]
test  = df[df['date'] >= df['date'].max() - pd.Timedelta(days=7)]

X_train = train.drop(['sales','date'], axis=1)
y_train = train['sales']
X_test  = test.drop(['sales','date'], axis=1)
y_test  = test['sales']


# ================================
# RANDOM FOREST MODEL
# ================================
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
print("RF RMSE =", rf_rmse)

feat_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)


# ================================
# XGBOOST MODEL
# ================================
xgb = XGBRegressor(n_estimators=300, learning_rate=0.08)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
print("XGB RMSE =", xgb_rmse)


# ================================
# ARIMA SAMPLE
# ================================
sample = df[(df.store==1) & (df.item==1)]
ts = sample.set_index('date')['sales']

arima_model = ARIMA(ts, order=(5,1,1)).fit()
arima_forecast = arima_model.forecast(steps=7)

print("ARIMA Forecast:")
print(arima_forecast)


# ================================
# ANOMALY DETECTION (Z-score)
# ================================
df['zscore'] = df.groupby(['store','item'])['sales'].transform(
    lambda x: (x - x.mean()) / x.std()
)
anomalies = df[np.abs(df['zscore']) > 3]

anomalies.to_csv("anomalies_detected.csv", index=False)
feat_imp.to_csv("feature_importance.csv", index=False)


# ================================
# SAVE FORECASTS FOR CHATBOT
# ================================
test['rf_pred'] = rf_preds
test['xgb_pred'] = xgb_preds
test.to_csv("precomputed_forecasts.csv", index=False)


# ================================
# RULE-BASED CHATBOT
# ================================
def chatbot(query):
    query = query.lower()

    if "predicted" in query or "forecast" in query:
        try:
            parts = query.split()
            store = int(parts[parts.index("store")+1])
            item  = int(parts[parts.index("item")+1])
            date  = pd.to_datetime(parts[-1])

            fc = pd.read_csv("precomputed_forecasts.csv")
            row = fc[(fc.store==store) & (fc.item==item) & (fc.date==date)]

            if len(row):
                return (
                    f"RF={row.rf_pred.values[0]:.2f}, "
                    f"XGB={row.xgb_pred.values[0]:.2f}"
                )
            else:
                return "No forecast available."
        except:
            return "Query format error."

    if "anomalies" in query:
        try:
            store = int(query.split("store")[1].split()[0])
            an = anomalies[anomalies.store==store]
            return f"Found {len(an)} anomalies."
        except:
            return "Error reading store ID."

    if "feature" in query:
        return str(feat_imp.head(5))

    return "I can answer: forecasts, anomalies, feature importance."


print("\nChatbot test:")
print(chatbot("predicted sales for store 1 item 1 on 2023-03-31"))
print(chatbot("highlight anomalies for store 2"))
print(chatbot("which features affect sales predictions?"))
