import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # For saving the model
import onnx
import onnxruntime as ort

# Load the dataset
df = pd.read_csv('dataset.csv')

# Handle missing values (optional)
df = df.dropna()  # Drop rows with missing values

# Features and target variable
X = df.drop('Total_egg_production', axis=1)
y = df['Total_egg_production']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)

# Check if accuracy (R-squared) is greater than 80%
if r2 > 0.8:
    print("Model achieved accuracy greater than 80%.")
else:
    print("Model accuracy is below 80%. Consider hyperparameter tuning or trying different models.")

# Save the model using joblib
joblib_file = 'model.pkl'
joblib.dump(model, joblib_file)
print(f"Model saved as {joblib_file}")

# Convert and save the model to ONNX format (if skl2onnx is available)
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [('float_input', FloatTensorType([None, X_train_scaled.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    onnx_file_path = 'model.onnx'
    with open(onnx_file_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX model saved to {onnx_file_path}")

except ImportError:
    print("sklearn-onnx is not available. ONNX conversion skipped.")

# Load the ONNX model and make predictions (if available)
if 'onnx_model' in locals():
    onnx_session = ort.InferenceSession(onnx_file_path)
    onnx_input_name = onnx_session.get_inputs()[0].name

    new_data = pd.DataFrame({
        'Amount_of_chicken': [2700],
        'Amount_of_Feeding': [185],
        'Ammonia': [14.9],
        'Temperature': [29.5],
        'Humidity': [50.7],
        'Light_Intensity': [317],
        'Noise': [200]
    })

    new_data_scaled = scaler.transform(new_data)
    onnx_predictions = onnx_session.run(None, {onnx_input_name: new_data_scaled.astype('float32')})
    print("Predicted Egg Production for new data (ONNX):", onnx_predictions[0][0])
