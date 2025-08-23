# Water Wise: A Farmer-Centric Classification System for Water Footprint

## Overview
**Water Wise** is a system designed to help farmers manage their water resources more efficiently and sustainably. The system provides practical tools to classify and assess water usage, helping farmers make informed decisions that lead to reduced waste, enhanced sustainability, and optimized costs. This solution is driven by machine learning and tailored to support farming practices with actionable insights and recommendations.

## Features
- **Water Classification:** Categorizes water footprints into Blue, Green, and Grey water based on farming practices.
- **ML-Powered Predictions:** Employs models like Random Forest, Gradient Boosting, and Support Vector Machines (SVM) for water usage and image analysis.
- **Crop Recommendations:** Suggests crops based on environmental factors such as latitude, longitude, soil type, and climate zones using an if-else decision ladder.
- **Interactive Interface:** Streamlit-based web application for an easy-to-use and engaging user experience.
- **Visualization:** Displays prediction probabilities and water distribution through interactive charts.

## Technologies Used
- **Programming Language:** Python
- **Machine Learning Models:** Random Forest, Gradient Boosting, SVM
- **Frontend Framework:** Streamlit
- **Visualization Libraries:** Matplotlib, Seaborn

## Installation

### Prerequisites
Ensure you have Python installed on your system. Install the required packages by running the following command:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn tensorflow sklearn
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/water-wise.git
cd water-wise
```

## Usage
1. Run the Streamlit application:
    ```bash
    streamlit run main.py
    ```

2. Open the application in your web browser at `http://localhost:8501`.

3. Log in or sign up to access the features:
   - **Tabular Analysis:** Input parameters like soil moisture, rainfall, and crop type to predict water usage categories.
   - **Image Analysis:** Upload farm images for water category predictions using trained image models.
   - **Farm Recommendations:** Enter location and soil type to receive tailored recommendations for crops and water management.

## Project Workflow
1. **Data Preprocessing:**
   - Utilize the `extended_farmland_water_dataset.csv` for tabular analysis.
   - Perform advanced preprocessing for feature engineering.

2. **Model Training:**
   - Train and evaluate multiple models to identify the best-performing one for water category predictions.

3. **Image Analysis:**
   - Use TensorFlow and SVM for image-based water category predictions.

4. **Recommendations:**
   - Use an if-else decision ladder to suggest crops based on geospatial and environmental factors.

## Key Components
### 1. Tabular Analysis
- Predict water categories (Blue, Green, Grey) using environmental and agricultural parameters.
- Visualize probabilities and water usage distribution.

### 2. Image Analysis
- Classify farm images into water categories using advanced ML models.
- Display results with probabilities.

### 3. Farm Recommendations
- Provide location-based crop suggestions.
- Highlight water management strategies based on climate zones and soil types.

## Future Enhancements
- Expand datasets for improved model accuracy.
- Integrate real-time weather API for dynamic recommendations.
- Add support for multilingual interfaces to cater to diverse user groups.

## Acknowledgments
- **Libraries and Frameworks:** Streamlit, TensorFlow, Scikit-learn
- **Inspiration:** Sustainable farming practices and resource management.

---
Start optimizing your farm's water usage with **Water Wise** today!
