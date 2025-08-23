import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
import sqlite3
import hashlib
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import xgboost as xgb
from lightgbm import LGBMClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Enhanced crop database with more detailed categories
CROP_DATABASE = {
    "tropical": {
        "primary": ["rice", "sugarcane", "banana"],
        "secondary": ["coconut", "coffee", "cassava"],
        "seasonal": ["pineapple", "mango", "papaya"]
    },
    "subtropical": {
        "primary": ["cotton", "citrus", "tobacco"],
        "secondary": ["corn", "soybean", "tea"],
        "seasonal": ["peach", "lychee", "longan"]
    },
    "temperate": {
        "primary": ["wheat", "barley", "potato"],
        "secondary": ["apple", "grape", "pear"],
        "seasonal": ["strawberry", "cherry", "plum"]
    },
    "arid": {
        "primary": ["dates", "millet", "sorghum"],
        "secondary": ["cotton", "chickpea", "safflower"],
        "seasonal": ["watermelon", "cantaloupe", "squash"]
    },
    "mediterranean": {
        "primary": ["olive", "grape", "wheat"],
        "secondary": ["citrus", "almond", "fig"],
        "seasonal": ["tomato", "eggplant", "pepper"]
    }
}

# Enhanced weather patterns with more parameters
CLIMATE_PATTERNS = {
    "tropical": {
        "avg_temp": (25, 35),
        "rainfall": (2000, 3000),
        "humidity": (70, 90),
        "soil_ph": (5.5, 6.5),
        "sunshine_hours": (10, 12),
        "wind_speed": (5, 15)
    },
    "subtropical": {
        "avg_temp": (20, 30),
        "rainfall": (1000, 2000),
        "humidity": (60, 80),
        "soil_ph": (6.0, 7.0),
        "sunshine_hours": (8, 10),
        "wind_speed": (10, 20)
    },
    "temperate": {
        "avg_temp": (10, 20),
        "rainfall": (750, 1500),
        "humidity": (50, 70),
        "soil_ph": (6.5, 7.5),
        "sunshine_hours": (6, 8),
        "wind_speed": (15, 25)
    },
    "arid": {
        "avg_temp": (20, 40),
        "rainfall": (200, 500),
        "humidity": (20, 40),
        "soil_ph": (7.0, 8.5),
        "sunshine_hours": (12, 14),
        "wind_speed": (20, 30)
    },
    "mediterranean": {
        "avg_temp": (15, 25),
        "rainfall": (500, 1000),
        "humidity": (40, 60),
        "soil_ph": (6.8, 7.8),
        "sunshine_hours": (8, 10),
        "wind_speed": (10, 20)
    }
}


# Soil characteristics database
SOIL_CHARACTERISTICS = {
    "sandy": {
        "water_retention": "low",
        "drainage": "high", 
        "nutrient_retention": "low",
        "aeration": "high",
        "ph_range": (5.5, 7.0),
        "organic_matter": (1, 3),
        "recommended_crops": ["carrots", "potatoes", "peanuts"]
    },
    "clay": {
        "water_retention": "high",
        "drainage": "low",
        "nutrient_retention": "high", 
        "aeration": "low",
        "ph_range": (6.0, 7.5),
        "organic_matter": (3, 5),
        "recommended_crops": ["wheat", "rice", "corn"]
    },
    "loam": {
        "water_retention": "medium",
        "drainage": "medium",
        "nutrient_retention": "medium",
        "aeration": "medium", 
        "ph_range": (6.0, 7.0),
        "organic_matter": (4, 6),
        "recommended_crops": ["vegetables", "fruits", "grains"]
    },
    "silt": {
        "water_retention": "medium-high",
        "drainage": "medium-low",
        "nutrient_retention": "medium-high",
        "aeration": "medium",
        "ph_range": (6.0, 7.0),
        "organic_matter": (3, 5),
        "recommended_crops": ["vegetables", "berries", "grass"]
    },
    "peat": {
        "water_retention": "very high",
        "drainage": "poor",
        "nutrient_retention": "high",
        "aeration": "low",
        "ph_range": (4.5, 6.0),
        "organic_matter": (30, 70),
        "recommended_crops": ["blueberries", "cranberries", "azaleas"]
    }
}

# Authentication class for managing users

class AuthManager:
    def __init__(self):
        self.init_db()
        # Initialize session state variables if not set
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
        if 'username' not in st.session_state:
            st.session_state.username = None

    def init_db(self):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (username TEXT PRIMARY KEY, 
                      password TEXT,
                      email TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def add_user(self, username, password, email):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password, email) VALUES (?,?,?)", 
                     (username, self.hash_password(password), email))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def verify_user(self, username, password):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        result = c.fetchone()
        conn.close()
        if result and result[0] == self.hash_password(password):
            return True
        return False

    def render_auth_ui(self):
        # Add signout button if logged in
        if st.session_state.logged_in:
            st.sidebar.title(f"Welcome, {st.session_state.username}!")
            if st.sidebar.button("Sign Out"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.rerun()  # Safely reruns the app
            return

        # Login section
        st.title("Login")
        login_col1, login_col2 = st.columns(2)
        
        with login_col1:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
                if self.verify_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Logged in successfully!")
                    st.rerun()  # Safely reruns the app
                else:
                    st.error("Invalid username or password")

        # Signup section  
        with login_col2:
            st.subheader("New User? Sign Up Here")
            with st.form("signup_form"):
                new_username = st.text_input("New Username", key="signup_username")
                new_password = st.text_input("New Password", type="password", key="signup_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
                email = st.text_input("Email", key="signup_email")
                signup = st.form_submit_button("Sign Up")
                
                if signup:
                    if new_password != confirm_password:
                        st.error("Passwords don't match!")
                    elif not new_username or not new_password or not email:
                        st.error("Please fill all fields!")
                    else:
                        if self.add_user(new_username, new_password, email):
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error("Username already exists!")


class WaterFootprintAnalyzer:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.preprocessor = None
        self.image_model = None

    def advanced_preprocessing(self):
        # Identify numeric and categorical columns
        numeric_features = [
            'soil_moisture', 'rainfall', 'temperature', 
            'irrigation_frequency', 'land_slope', 
            'organic_matter_content'
        ]
        categorical_features = ['crop_type']

        # Create preprocessor with advanced transformations
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        # Prepare features and target
        self.X = self.dataset.drop('water_category', axis=1)
        self.y = self.dataset['water_category']

        # Split the data with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=0.2, 
            random_state=42, 
            stratify=self.y
        )

        return self.preprocessor

    def train_multiple_models(self):
        # Create a list of models to try
        models = [
            ('Random Forest', RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )),
            ('Gradient Boosting', GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )),
            ('SVM', SVC(
                kernel='rbf', 
                probability=True, 
                C=1.0, 
                random_state=42
            ))
        ]

        # Best model storage
        best_model = None
        best_accuracy = 0

        # Evaluate and store results
        results = []

        for name, model in models:
            # Create a pipeline with preprocessing and model
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])

            # Fit the model
            pipeline.fit(self.X_train, self.y_train)

            # Evaluate the model
            accuracy = pipeline.score(self.X_test, self.y_test)
            
            # Predictions for detailed metrics
            y_pred = pipeline.predict(self.X_test)
            
            # Compute detailed classification report
            report = classification_report(self.y_test, y_pred, output_dict=True)
            
            # Store results
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': report['weighted avg']['precision'],
                'Recall': report['weighted avg']['recall'],
                'F1-Score': report['weighted avg']['f1-score']
            })

            # Track the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.model = pipeline

        return pd.DataFrame(results), self.model

    def detailed_model_analysis(self):
        # Detailed predictions and probabilities
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Compute average confidence
        predicted_classes = np.argmax(y_prob, axis=1)
        confidence_levels = y_prob[np.arange(len(predicted_classes)), predicted_classes]
        avg_confidence = np.mean(confidence_levels) * 100

        return {
            'confusion_matrix': cm,
            'avg_confidence': avg_confidence,
            'classification_report': classification_report(self.y_test, y_pred)
        }

    def predict_water_category(self, sample_data):
        # Convert sample data to DataFrame
        sample_df = pd.DataFrame([sample_data])
        
        # Predict probabilities
        probabilities = self.model.predict_proba(sample_df)[0]
        predicted_class = np.argmax(probabilities)

        return predicted_class, probabilities

    def load_image_model(self):
        # Load the pre-trained ResNet50 model with custom top layers for classification
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model = Sequential([
            base_model,
            Flatten(),
            Dense(256, activation='relu'),
            Dense(3, activation='softmax')  # Blue, Green, Grey Water categories
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.image_model = model
        return model

    def predict_image_category(self, uploaded_image):
        if self.image_model is None:
            st.error("Image model not loaded!")
            return None

        try:
            # Preprocess the image
            image = Image.open(uploaded_image)
            image = image.resize((224, 224))  # Resize image to fit model input size
            image = np.array(image) / 255.0  # Normalize image
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Use the model to predict the class
            predictions = self.image_model.predict(preprocess_input(image))
            predicted_class = np.argmax(predictions, axis=1)

            # Mapping indices to categories
            categories = ["Blue Water", "Green Water", "Grey Water"]
            predicted_category = categories[predicted_class[0]]

            return predicted_category, predictions[0]  # Returning category and probabilities

        except Exception as e:
            st.error(f"Error during image prediction: {str(e)}")
            return None, None


def determine_climate_zone(latitude):
    """Determine climate zone based on latitude"""
    if -23.5 <= latitude <= 23.5:
        return "tropical"
    elif (23.5 < latitude <= 35) or (-35 <= latitude < -23.5):
        return "subtropical"
    elif (35 < latitude <= 50) or (-50 <= latitude < -35):
        return "temperate"
    elif (latitude > 50) or (latitude < -50):
        return "arid"
    else:
        return "mediterranean"

def get_location_based_crops(latitude, longitude):
    """Get suitable crops based on location"""
    climate_zone = determine_climate_zone(latitude)
    return CROP_DATABASE[climate_zone]

def get_weather_data(latitude, longitude):
    """Simulate weather data based on location"""
    climate_zone = determine_climate_zone(latitude)
    patterns = CLIMATE_PATTERNS[climate_zone]
    
    current_temp = np.random.uniform(patterns["avg_temp"][0], patterns["avg_temp"][1])
    current_rainfall = np.random.uniform(patterns["rainfall"][0], patterns["rainfall"][1])
    current_humidity = np.random.uniform(patterns["humidity"][0], patterns["humidity"][1])
    
    return {
        "temperature": round(current_temp, 1),
        "rainfall": round(current_rainfall, 1),
        "humidity": round(current_humidity, 1),
        "climate_zone": climate_zone
    }

def main():
    st.set_page_config(page_title="Water Wise: A Farmer-Centric Classification System for Water Footprint", page_icon="💧", layout="wide")
    
    # Initialize session state variables if not already set
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'page' not in st.session_state:
        st.session_state.page = 'login'

    auth_manager = AuthManager()

    # Show logout button if logged in
    if st.session_state.logged_in:
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.page = 'login'
            st.rerun()

        st.title("💧 Water Wise: A Farmer-Centric Classification System for Water Footprint")
        st.markdown("""
        ### Comprehensive Water Category Prediction
        This tool helps analyze water usage categories based on various environmental and agricultural parameters.
        """)

        # Navigation for logged in users
        page = st.sidebar.selectbox("Select Analysis Type", 
                                  ["Tabular Analysis", "Image Analysis", "Farm Recommendations"])
        
        if page == "Tabular Analysis":
            # Load and run tabular analysis code
            analyzer = WaterFootprintAnalyzer('extended_farmland_water_dataset.csv')
            analyzer.advanced_preprocessing()
            model_results, best_model = analyzer.train_multiple_models()
            
            st.subheader("Water Usage Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Environmental Factors")
                soil_moisture = st.slider('Soil Moisture', 0.3, 0.9, 0.6, step=0.01)
                rainfall_mm = st.number_input('Rainfall (mm per hectare)', 500, 1500, 750)
                temperature = st.number_input('Temperature (°C)', 20.0, 35.0, 28.5, step=0.1)
            
            with col2:
                st.subheader("Agricultural Parameters")
                crop_type = st.selectbox('Crop Type', ['wheat', 'rice', 'corn', 'soybean', 'cotton'])
                irrigation_frequency = st.number_input('Irrigation Frequency', 1, 5, 3)
                land_slope = st.number_input('Land Slope', 0.0, 0.2, 0.05, step=0.01)
                organic_matter = st.number_input('Organic Matter Content', 1.0, 5.0, 3.2, step=0.1)
            
            hectares = st.number_input('Land Area (Hectares)', min_value=1, max_value=100, value=10)

            if st.button('Analyze'):
                sample_data = {
                    'soil_moisture': soil_moisture,
                    'rainfall': rainfall_mm,
                    'temperature': temperature,
                    'crop_type': crop_type,
                    'irrigation_frequency': irrigation_frequency,
                    'land_slope': land_slope,
                    'organic_matter_content': organic_matter
                }

                predicted_class, probabilities = analyzer.predict_water_category(sample_data)
                categories = ["Blue Water", "Green Water", "Grey Water"]

                st.success(f'Predicted Water Category: {categories[predicted_class]}')

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(categories, probabilities, color=['blue', 'green', 'grey'])
                ax.set_ylabel('Probability')
                ax.set_title('Water Category Prediction Probabilities')
                st.pyplot(fig)

                rainfall_liters = rainfall_mm * 10000 * hectares
                irrigation_liters = rainfall_liters * irrigation_frequency * 0.2

                fig, ax = plt.subplots(figsize=(10, 6))
                rainfall_data = {
                    'Rainfall Needed': rainfall_liters,
                    'Additional Irrigation': irrigation_liters,
                }
                
                ax.bar(rainfall_data.keys(), rainfall_data.values(), color=['skyblue', 'lightgreen'])
                ax.set_title(f"Total Water Distribution for {hectares} Hectares")
                ax.set_ylabel("Water Volume (Liters)")
                
                for i, (key, value) in enumerate(rainfall_data.items()):
                    ax.text(i, value, f'{value:,.0f}', ha='center', va='bottom')
                
                st.pyplot(fig)

                st.markdown(f"""
                ### Water Usage Analysis
                - Total Rainfall Volume: {rainfall_liters:,.0f} liters
                - Additional Irrigation Needed: {irrigation_liters:,.0f} liters
                - Total Water Requirement: {rainfall_liters + irrigation_liters:,.0f} liters
                """)

        elif page == "Image Analysis":
            analyzer = WaterFootprintAnalyzer('extended_farmland_water_dataset.csv')
            analyzer.load_image_model()
            
            st.subheader("Image-based Water Analysis")
            uploaded_image = st.file_uploader("Upload Farm Image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_image:
                predicted_category, probabilities = analyzer.predict_image_category(uploaded_image)
                if predicted_category:
                    st.image(uploaded_image, caption="Analyzed Image", use_container_width=True)
                    st.success(f"Predicted Water Category: {predicted_category}")
                    
                    categories = ['Blue Water', 'Green Water', 'Grey Water']
                    fig, ax = plt.subplots()
                    ax.bar(categories, probabilities, color=['blue', 'green', 'grey'])
                    ax.set_title('Water Category Probabilities')
                    st.pyplot(fig)

        elif page == "Farm Recommendations":
            st.subheader("Location-based Farm Recommendations")
            
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input("Latitude", -90.0, 90.0, 0.0)
                longitude = st.number_input("Longitude", -180.0, 180.0, 0.0)
            with col2:
                soil_type = st.selectbox("Soil Type", ["sandy", "clay", "loam"])
            
            if st.button("Get Recommendations"):
                weather_data = get_weather_data(latitude, longitude)
                suitable_crops = get_location_based_crops(latitude, longitude)
                
                st.subheader("Climate Analysis")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Temperature", f"{weather_data['temperature']}°C")
                with col2:
                    st.metric("Rainfall", f"{weather_data['rainfall']} mm")
                with col3:
                    st.metric("Humidity", f"{weather_data['humidity']}%")
                
                st.subheader("Recommended Crops")
                st.write(suitable_crops)
                
                st.subheader("Water Management")
                climate_zone = weather_data["climate_zone"]
                if climate_zone in ["arid"]:
                    st.info("Water conservation measures highly recommended, subsurface irrigation")
                elif climate_zone in ["mediterranean"]:
                    st.info("Water conservation measures highly recommended, subsurface irrigation")
                elif climate_zone in ["tropical"]:
                    st.info("Drip and sprinkler irrigation systems")
                
                st.markdown(f"""
                ### Soil Management Tips for {soil_type.title()} Soil
                - pH monitoring recommended
                - Regular organic matter addition
                - Proper drainage maintenance
                """)

    else:
        # Handle login/signup pages
        if st.session_state.page == 'login':
            st.title("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Login"):
                    if auth_manager.verify_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
            
            with col2:
                if st.button("Go to Signup"):
                    st.session_state.page = 'signup'
                    st.rerun()

        elif st.session_state.page == 'signup':
            st.title("Sign Up")
            with st.form("signup_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                email = st.text_input("Email")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    signup = st.form_submit_button("Sign Up")
                    if signup:
                        if new_password != confirm_password:
                            st.error("Passwords don't match!")
                        elif not new_username or not new_password or not email:
                            st.error("Please fill all fields!")
                        else:
                            if auth_manager.add_user(new_username, new_password, email):
                                st.success("Account created successfully!")
                                st.session_state.page = 'login'
                                st.rerun()
                            else:
                                st.error("Username already exists!")
                
                with col2:
                    if st.form_submit_button("Back to Login"):
                        st.session_state.page = 'login'
                        st.rerun()

if __name__ == "__main__":
    main()

