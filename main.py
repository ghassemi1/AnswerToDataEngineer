
from pickle import dump
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from Functions import read_files, check_num_none_and_inf, save_dataframe

# Problem 1: Raw Data Processing
# Problem 2: Feature Engineering

# read 40 first files from "etfs" and "stock" folders
df = read_files(main_path="../kaggle", num_files=40)

# check and print number of nan and inf values
check_num_none_and_inf(df)

# drop nan values
df.dropna(inplace=True)

# select specific features
data = df[['vol_moving_avg', 'adj_close_rolling_med', 'Volume']]

# save dataset in a specific directory
save_dataframe(data, main_path="./dataframe", file_name='data',
               save_type='parquet')  # csv or parquet type files


# separate train and test dataset
dataset_train = data[:int(data.shape[0]*0.8)]
dataset_test = data[int(data.shape[0]*0.8):]

X_train, y_train = dataset_train.iloc[:, :-1], dataset_train.iloc[:, -1]
X_test, y_test = dataset_test.iloc[:, :-1], dataset_test.iloc[:, -1]

# Problem 3: Integrate ML Training
# Create a XGBRegressor model
model = XGBRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)
# y_pred = scaler.inverse_transform(y_pred)

# # Calculate the variance
print('For XGBRegressor model: ')
print(f"mean_absolute_error: {mean_absolute_error(y_test, y_pred)}")
print(f"mean_squared_error: {mean_squared_error(y_test, y_pred)}")
print(f"metrics.r2_score: {metrics.r2_score(y_test, y_pred)}")
print('metrics.explained_variance_score : ',
      metrics.explained_variance_score(y_test, y_pred))
print('metrics.mean_squared_error : ', metrics.r2_score(y_test, y_pred))
print('metrics.explained_variance_score : ',
      metrics.explained_variance_score(y_test, y_pred))

# Save as a model
with open('./heroku/models/model.pkl', 'wb') as f:
    dump(model, f)
