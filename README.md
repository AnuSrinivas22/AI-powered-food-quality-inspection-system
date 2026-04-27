# AI-powered-food-quality-inspection-system
AI-powered food quality inspection system using YOLOv8 and deep learning to detect, classify, and score freshness of fruits with defect analysis and decision support.

Food Quality Inspection System using Computer Vision
 Overview

This project presents an AI-powered Food Quality Inspection System that automatically evaluates the freshness of fruits using computer vision techniques.

The system detects food items, analyzes their visual features, and classifies them into quality categories such as Fresh or Rotten, along with a quality score and decision (Accept/Reject).

🚀 Features
🔍 Object Detection using YOLOv8
🧠 Quality Classification using Deep Learning (MobileNetV2)
🎯 Defect Detection using OpenCV
📊 Quality Scoring System (multi-factor evaluation)
📷 Supports image input (extendable to video/live feed)
🌐 Streamlit UI for easy interaction
🧠 Problem Statement

In industries like agriculture and food processing, quality inspection is often:

Manual
Time-consuming
Inconsistent

This project automates the process to improve:

Accuracy
Speed
Scalability
💡 Solution Approach

The system follows a multi-stage pipeline:

Input Image
YOLOv8 detects the food object
CNN model classifies freshness
OpenCV detects defects (spots, spoilage)
Quality score is calculated
Final decision: Accept / Reject
🏗️ Tech Stack
Python
TensorFlow / Keras
OpenCV
YOLOv8 (Ultralytics)
Streamlit
NumPy, Matplotlib
📂 Dataset

The dataset contains images of:

🍎 Apple
🍌 Banana
🍓 Strawberry
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/01b5853a-48f0-49e8-b939-8bb177081dd5" />


Each categorized into:

Fresh
Rotten
Dataset Structure (after preprocessing)
dataset/
   train/
      fresh/
      rotten/
⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/food-quality-inspection.git
cd food-quality-inspection
2. Install dependencies
pip install -r requirements.txt
▶️ Usage
🔹 Run Jupyter Notebook
jupyter notebook
🔹 Run Streamlit App
streamlit run app.py
🧪 Model Training
Used MobileNetV2 (Transfer Learning)
Image size: 224x224
Data augmentation applied
Validation split: 20%
📊 Output Example
Object: Banana  
Quality: Rotten  
Defects: High Spots Detected  
Score: 52/100  
Decision: REJECT  
🎯 Applications
🏭 Food Industry Quality Control
🛒 Automated Sorting Systems
🚜 Agriculture Monitoring
🧾 Supply Chain Optimization
🔮 Future Improvements
Add more food categories
Real-time video processing
Deploy on edge devices (Raspberry Pi)
Improve defect detection using segmentation
Custom YOLO training for specific fruits
🤝 Contribution

Contributions are welcome!
Feel free to fork the repo and submit pull requests.

Author:
Anushree S

👨‍💻 Author

Your Name
