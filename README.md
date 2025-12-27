# 🏃 Human Activity Recognition using IMU Sensors & LSTM

A real-time Human Activity Recognition (HAR) system using IMU sensor data
(accelerometer & gyroscope) and an LSTM deep learning model, deployed with Streamlit.

## 🚀 Features
- LSTM-based activity classification
- IMU sensor fusion (ax, ay, az, gx, gy, gz)
- Real-time & CSV-based prediction
- Streamlit web interface
- Healthcare & fitness use case

## 🧠 Activities Recognized
1. Walking Upstairs  
2. Walking Downstairs  
3. Walking  
4. Sitting  
5. Standing  
6. Jogging  

## 📂 Project Structure

#HAR_Streamlit_App/

├── src/

│ ├── data_loader.py

│ ├── preprocessing.py

│ ├── model_builder.py

│ ├── train.py

│ └── predict.py

├── app.py

├── requirements.txt

├── README.md



## ⚙️ Installation
```bash
pip install -r requirements.txt


python src/train.py

streamlit run app.py

```bash


 Dataset

IMU-based Human Activity Recognition Dataset
Publicly available, multi-sensor time-series data.

 Model

LSTM (2 layers)

Time window: 50 samples

Optimizer: Adam

Loss: Sparse Categorical Crossentropy



