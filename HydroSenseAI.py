import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Simulated Data Collection
n_samples = 5000

# Simulating environmental variables
temp = np.random.uniform(20, 45, n_samples)  # Temperature in Celsius
humidity = np.random.uniform(20, 80, n_samples)  # Relative Humidity (%)
soil_moisture = np.random.uniform(10, 90, n_samples)  # Soil moisture level (%)
crop_type = np.random.choice(["Wheat", "Rice", "Sugarcane"], n_samples)  # Crop type
rainfall = np.random.uniform(0, 100, n_samples)  # Rainfall (mm)

# Improved Water Need Calculation (Based on FAO Irrigation Guidelines)
def calculate_irrigation(temp, humidity, soil_moisture, crop, rainfall):
    base_water_need = {"Wheat": 500, "Rice": 1200, "Sugarcane": 1500}  # mm/season
    
    # Temperature Factor (FAO crop coefficient adjustments)
    temp_factor = 1 + (temp - 30) * 0.02  # Each 1°C above 30 increases need by 2%
    
    # Humidity Factor (Inverse relation with water need)
    humidity_factor = 1 - (humidity - 50) * 0.01  # Each 1% above 50 reduces need by 1%
    
    # Soil Moisture Factor (Higher soil moisture reduces irrigation need)
    soil_factor = 1 - (soil_moisture / 100)
    
    # Final Calculation with Rainfall Compensation
    water_need = base_water_need[crop] * temp_factor * humidity_factor * soil_factor - (rainfall * 0.7)
    return max(0, water_need)  # Ensure no negative values

# Compute Water Need
water_needed = np.array([calculate_irrigation(temp[i], humidity[i], soil_moisture[i], crop_type[i], rainfall[i]) for i in range(n_samples)])

# Creating DataFrame
data = pd.DataFrame({
    'Temperature': temp,
    'Humidity': humidity,
    'Soil_Moisture': soil_moisture,
    'Crop_Type': crop_type,
    'Rainfall': rainfall,
    'Water_Needed': water_needed
})

# One-Hot Encoding Crop Type
data = pd.get_dummies(data, columns=['Crop_Type'], drop_first=True)

# Splitting Data
X = data.drop(columns=['Water_Needed'])
y = data['Water_Needed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (MinMax works better for environmental data)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Improved AI Model
model = keras.Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile Model
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])

# Learning Rate Reduction Callback
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

# Train Model
history = model.fit(X_train_scaled, y_train, epochs=30, validation_data=(X_test_scaled, y_test), verbose=1, callbacks=[lr_scheduler])

# Evaluate Model
loss, mae = model.evaluate(X_test_scaled, y_test)
print(f"Model Test Loss: {loss:.4f}, Mean Absolute Error: {mae:.4f}")

# Plot Training Progress
plt.figure(figsize=(10, 5))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('Training Progress')
plt.show()

# Improved Actual Water Use Simulation (Simulates Real Irrigation Inefficiencies)
manual_irrigation_variation = np.random.uniform(1.1, 1.3, len(y_test))  # Farmers over-irrigate by 10-30%
actual_water_use = manual_irrigation_variation * y_test

# AI-Predicted Water Use
optimized_water_use = model.predict(X_test_scaled).flatten()

# Comparing Results
improvement = ((actual_water_use - optimized_water_use) / actual_water_use) * 100
average_savings = np.mean(improvement)
print(f"Average Water Savings with AI Optimization: {average_savings:.2f}%")

# Visualizing Improvement
plt.figure(figsize=(10, 5))
sns.histplot(improvement, bins=30, kde=True)
plt.xlabel('Water Savings (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Water Savings with AI Optimization')
plt.xlim(-50, 100)
plt.show()
