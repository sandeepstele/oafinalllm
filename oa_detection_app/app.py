import os
from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import pandas as pd
import joblib
import xgboost 
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics._scorer
if not hasattr(sklearn.metrics._scorer, '_passthrough_scorer'):
    # You can set it to a dummy function or None. Here we set it to a lambda.
    sklearn.metrics._scorer._passthrough_scorer = lambda *args, **kwargs: None
from io import BytesIO




app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# ----- Load your models (they already exist in the models folder) -----
# Multi-class models (Clinical + X-ray fusion)
multi_tf_model = tf.keras.models.load_model(os.path.join('models', 'weightsfinal.h5'))
multi_xgb_model = joblib.load(os.path.join('models', 'xgb_clf_model_multi.pkl'))

# Binary models (for binary OA detection)
binary_tf_model = tf.keras.models.load_model(os.path.join('models',  'weightb.h5'))
binary_xgb_model = joblib.load(os.path.join('models', 'xgb_classifier.pkl'))

# ----- Default Clinical Data & Preprocessing -----
default_sample_data = np.array([[59, 181.1, 78.1, 90.9, 23.8, 5, 1, 5, 3, 0, 0, 1, 1, 100, 2, 3, 2, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0]])
clinical_columns = [
    "AGE", "HEIGHT", "WEIGHT", "MAX WEIGHT", "BMI", "FREQUENT PAIN", "SURGERY",
    "RISK", "SXKOA", "SWELLING", "BENDING FULLY", "SYMPTOMATIC", "CREPITUS",
    "KOOS PAIN SCORE", "osteophytes_y", "jsn_y", "osfl", "scfl", "cyfl",
    "ostm", "sctm", "cytm", "attm", "osfm", "scfm", "cyfm", "ostl", "sctl",
    "cytl", "attl"
]
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

def preprocess_clinical_data(data_array):
    df_sample = pd.DataFrame(data_array, columns=clinical_columns)
    interaction_features = poly.fit_transform(df_sample)
    interaction_df = pd.DataFrame(interaction_features, columns=poly.get_feature_names_out(df_sample.columns))
    df_out = pd.concat([df_sample, interaction_df], axis=1)
    return df_out

# ----- Image Preprocessing -----
def preprocess_image_file(img_file):
    # Read the FileStorage object into a BytesIO stream
    img_stream = BytesIO(img_file.read())
    # Reset the pointer so the file can be used again if needed
    img_file.seek(0)
    # Load the image from the BytesIO stream with the desired target size
    img = keras_image.load_img(img_stream, target_size=(300, 300))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----- Routes -----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clinical')
def clinical():
    return render_template('clinical.html')

@app.route('/xray')
def xray():
    return render_template('xray.html')

@app.route('/fusion', methods=['GET', 'POST'])
def fusion():
    result_text = ""
    if request.method == 'POST':
        model_type = request.form.get('model_type')
        if 'image_file' not in request.files:
            flash("No image file provided.")
            return redirect(request.url)
        file = request.files['image_file']
        if file.filename == '':
            flash("No file selected.")
            return redirect(request.url)
        # Preprocess image and clinical data
        img_array = preprocess_image_file(file)
        clinical_df = preprocess_clinical_data(default_sample_data)
        
        if model_type == 'multi':
            predictions_tf = multi_tf_model.predict(img_array)
            # Convert clinical_df to numpy array
            xgb_probs = multi_xgb_model.predict_proba(clinical_df.values)
            w1 = 0.6530612244897959
            w2 = 0.34693877551020413
            fused_probs = (w1 * predictions_tf) + (w2 * xgb_probs)
            final_label = np.argmax(fused_probs, axis=1)[0]
            if final_label == 0:
                result_text = "Multi-class Prediction: OA not detected (0)."
            else:
                result_text = f"Multi-class Prediction: OA detected (1) with a KL score of {final_label}."
        elif model_type == 'binary':
            predictions_tf = binary_tf_model.predict(img_array)
            xgb_probs = binary_xgb_model.predict_proba(clinical_df.values)
            w1 = 0.4897959183673469
            w2 = 0.5102040816326531
            fused_probs = (w1 * predictions_tf) + (w2 * xgb_probs)
            final_label = np.argmax(fused_probs, axis=1)[0]
            result_text = f"Binary Prediction: OA Detection = {'Yes (1)' if final_label == 1 else 'No (0)'}."
        else:
            flash("Invalid model type selected.")
            return redirect(request.url)
        
        return render_template('fusion.html', result=result_text)
    return render_template('fusion.html')

if __name__ == '__main__':
    app.run(debug=True)
