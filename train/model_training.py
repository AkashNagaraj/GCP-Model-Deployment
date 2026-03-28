import pandas as pd 
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump, load

le = LabelEncoder()

df = pd.read_csv("global_ecommerce_forecasting.csv")
df["country"] = le.fit_transform(df["country"])
X = df[["year", "month", "week_of_year", "day_of_week", "order_hour", "is_weekend", "country"]]
y = df["sales_amount_gbp"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)

dump(model, "model.joblib")