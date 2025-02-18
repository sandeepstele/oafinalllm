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

@app.route('/clinical', methods=['GET', 'POST'])
def clinical():
    # Create a dictionary of default values from default_sample_data
    default_clinical_dict = dict(zip(clinical_columns, default_sample_data[0]))
    result_text = ""
    if request.method == 'POST':
        clinical_inputs = []
        # Loop through each clinical parameter; use default if input is empty.
        for col in clinical_columns:
            value = request.form.get(col)
            if value is None or value.strip() == '':
                clinical_inputs.append(default_clinical_dict[col])
            else:
                try:
                    clinical_inputs.append(float(value))
                except ValueError:
                    clinical_inputs.append(default_clinical_dict[col])
        clinical_array = np.array([clinical_inputs])
        processed_df = preprocess_clinical_data(clinical_array)
        
        # Generate prediction from clinical data using the XGBoost model.
        clinical_prediction_proba = multi_xgb_model.predict_proba(processed_df.values)
        final_label = np.argmax(clinical_prediction_proba, axis=1)[0]
        if final_label == 0:
            prediction_result = "Clinical Prediction: OA not detected (0)."
        else:
            prediction_result = f"Clinical Prediction: OA detected (1) with a KL score of {final_label}."
            
        result_text = (
            "Processed Clinical Data:<br>" +
            processed_df.to_html(classes="table table-bordered") +
            "<br><br>" + prediction_result
        )
        
        # Log the result to Flask logger and terminal.
        app.logger.info("Clinical Data Processed: %s", result_text)
        print("Clinical Data Processed:", result_text)
        
        return render_template('clinical.html', result=result_text, defaults=default_clinical_dict, columns=clinical_columns)
    return render_template('clinical.html', defaults=default_clinical_dict, columns=clinical_columns)
@app.route('/xray', methods=['GET', 'POST'])
def xray():
    result_text = ""
    if request.method == 'POST':
        if 'image_file' not in request.files:
            flash("No image file provided.")
            return redirect(request.url)
        file = request.files['image_file']
        if file.filename == '':
            flash("No file selected.")
            return redirect(request.url)
        # Preprocess the uploaded image using our BytesIO-based function
        img_array = preprocess_image_file(file)
        # Use the x-ray model (using the multi_tf_model for image-only prediction)
        predictions = multi_tf_model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=-1)[0]
        # Interpret the prediction (assuming 0 means "OA not detected" and >0 means "OA detected with KL score")
        if predicted_class == 0:
            result_text = "X-ray Prediction: OA not detected (0)."
        else:
            result_text = f"X-ray Prediction: OA detected (1) with a KL score of {predicted_class}."
        return render_template('xray.html', result=result_text)
    return render_template('xray.html')


@app.route('/fusion', methods=['GET', 'POST'])
def fusion():
    # Create a dictionary of default clinical values
    default_clinical_dict = dict(zip(clinical_columns, default_sample_data[0]))
    result_text = ""
    if request.method == 'POST':
        # Process clinical inputs: loop over each clinical parameter
        clinical_inputs = []
        for col in clinical_columns:
            value = request.form.get(col)
            if value is None or value.strip() == '':
                clinical_inputs.append(default_clinical_dict[col])
            else:
                try:
                    clinical_inputs.append(float(value))
                except ValueError:
                    clinical_inputs.append(default_clinical_dict[col])
        clinical_array = np.array([clinical_inputs])
        processed_df = preprocess_clinical_data(clinical_array)
        
        # Process the uploaded X-ray image
        if 'image_file' not in request.files:
            flash("No image file provided.")
            return redirect(request.url)
        file = request.files['image_file']
        if file.filename == '':
            flash("No file selected.")
            return redirect(request.url)
        img_array = preprocess_image_file(file)
        
        # --- Multi-class Prediction (interpreted as KL score) ---
        predictions_tf_multi = multi_tf_model.predict(img_array)
        xgb_probs_multi = multi_xgb_model.predict_proba(processed_df.values)
        w1_multi = 0.6530612244897959
        w2_multi = 0.34693877551020413
        fused_probs_multi = (w1_multi * predictions_tf_multi) + (w2_multi * xgb_probs_multi)
        final_label_multi = np.argmax(fused_probs_multi, axis=1)[0]
        if final_label_multi == 0:
            multi_result = "Multi-class Prediction: OA not detected (0)."
        else:
            multi_result = f"Multi-class Prediction: OA detected (1) with a KL score of {final_label_multi}."
        
        # --- Binary Prediction ---
        predictions_tf_binary = binary_tf_model.predict(img_array)
        xgb_probs_binary = binary_xgb_model.predict_proba(processed_df.values)
        w1_binary = 0.4897959183673469
        w2_binary = 0.5102040816326531
        fused_probs_binary = (w1_binary * predictions_tf_binary) + (w2_binary * xgb_probs_binary)
        final_label_binary = np.argmax(fused_probs_binary, axis=1)[0]
        binary_result = f"Binary Prediction: OA Detection = {'Yes (1)' if final_label_binary == 1 else 'No (0)'}."
        
        # Combine the results with the processed clinical data table
        result_text = (
            "Processed Clinical Data:<br>" +
            processed_df.to_html(classes="table table-bordered") +
            "<br><br>" +
            multi_result +
            "<br>" +
            binary_result
        )
        app.logger.info("Fusion result: %s", result_text)
        print("Fusion result:", result_text)
        return render_template('fusion.html', result=result_text, defaults=default_clinical_dict, columns=clinical_columns)
    return render_template('fusion.html', defaults=default_clinical_dict, columns=clinical_columns)
if __name__ == '__main__':
    app.run(debug=True)
