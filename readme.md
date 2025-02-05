# **HydroSenseAI: AI-Powered Smart Irrigation System**

## **Overview**

This project develops an AI-based model to optimize irrigation water usage for different crops. By analyzing environmental factors such as temperature, humidity, soil moisture, and rainfall, the model predicts the optimal amount of water needed, reducing over-irrigation and improving water efficiency.

## **Features**

- **Simulated Dataset**: Generates synthetic environmental data for three crop types (Wheat, Rice, Sugarcane).
- **AI Model**: A deep neural network predicts the required irrigation based on environmental conditions.
- **Efficiency Analysis**: Compares traditional irrigation methods with AI-optimized recommendations.
- **Visualization**: Displays training performance and water savings distribution.

## **Dataset**

The dataset consists of 5000 samples with the following features:

- Temperature (Celsius)
- Humidity (%)
- Soil Moisture (%)
- Crop Type (Wheat, Rice, Sugarcane)
- Rainfall (mm)
- Water Needed (computed based on FAO guidelines)

## **Model Architecture**

- Input: Environmental parameters
- Layers:
  - Dense (128 neurons, ReLU activation)
  - Batch Normalization
  - Dropout (20%)
  - Dense (64 neurons, ReLU activation)
  - Batch Normalization
  - Dropout (20%)
  - Dense (32 neurons, ReLU activation)
  - Dense (1 neuron, output layer)
- Optimizer: RMSprop
- Loss Function: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)

## **Installation**

1. Clone the repository:

```git clone https://github.com/KirolousFouty/HydroSenseAI```

cd ai-irrigation

2. Install dependencies:

```pip install numpy pandas tensorflow scikit-learn matplotlib seaborn```

3. Run the script:

```python HydroSenseAI.py```

## **Results**

- **Model Evaluation**:
  - Test Loss: Displayed in the output
  - Mean Absolute Error (MAE): Displayed in the output
- **Water Savings**:
  - The AI model optimizes irrigation needs compared to traditional methods.
  - The savings distribution is plotted as a histogram.

## **Future Improvements**

- Integrate real-world sensor data instead of synthetic data.
- Enhance model accuracy with advanced hyperparameter tuning.
- Deploy as a web or mobile application for farmers.

## **Author**

Kirolous Fouty, kirolous_fouty@aucegypt.edu