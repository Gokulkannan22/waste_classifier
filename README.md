# 🗑️ Waste Classifier using PyTorch, GradCAM & Streamlit

This project is a deep learning-based waste classification system that classifies images into **five types of waste**:
- 📦 Plastic
- 🗞 Paper
- 🍃 Organic
- 🔩 Metal
- 💻 E-waste (currently includes **batteries only**)
  
It uses:
- ✅ PyTorch for training
- ✅ GradCAM for model explainability
- ✅ Streamlit for the web app

---

## 📁 Project Structure
waste_classifier/
├── dataset/ # Input images
├── app.py # Streamlit app
├── final_waste_classifier.ipynb # Jupyter Notebook for training & analysis\
├── final_waste_model.pth
├── gradcam_utils.py # GradCAM helper functions
├── split_dataset.py # Train-validation split script
├── requirements.txt # Required libraries
└── README.md # Project info


Download the Trained Model File
The trained model file final_waste_model.pth is not included in the repository due to file size limits.

➡️ Download it from Google Drive:
[[📥 https://drive.google.com/file/d/1PVjwZ5Qgl3bGVSJDMtiTQOFoMjmjfu_X/view?usp=sharing](https://drive.google.com/file/d/1PVjwZ5Qgl3bGVSJDMtiTQOFoMjmjfu_X/view?usp=drive_link)](https://drive.google.com/file/d/1PVjwZ5Qgl3bGVSJDMtiTQOFoMjmjfu_X/view?usp=sharing)


Model Overview
🔍 Architecture: ResNet18 (Transfer Learning)

📈 Accuracy: 92.2% on validation set

🔥 Explainability: GradCAM visualizations

📋 Evaluation: Classification report, confusion matrix

🙋 Author
Gokul

🌐 GitHub: [Gokulkannan22](https://github.com/Gokulkannan22)


📜 License
This project is licensed under the MIT License.
