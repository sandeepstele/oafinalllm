import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables at the very start

import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import pandas as pd
import joblib
import xgboost
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics._scorer
if not hasattr(sklearn.metrics._scorer, '_passthrough_scorer'):
    sklearn.metrics._scorer._passthrough_scorer = lambda *args, **kwargs: None
from io import BytesIO
import json
import faiss
import openai
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

# Set up the OpenAI client with environment variables or hard-coded values for testing.
# For testing, we hard-code the token; in production, use environment variables.
openai.api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6InBhcmFkb3hAc3R1ZHkuaWl0bS5hYy5pbiJ9.YSy04n-k8bspy7aoR3eMxkBS2JWkMq91qJiF-2Op-vg"
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# ----- Define file paths for the FAISS index and document chunks -----
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "document_chunks.json"

# Global variables for FAISS index and cached chunks.
faiss_index = None
cached_chunks = None

def load_saved_index():
    global faiss_index, cached_chunks
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        faiss_index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            cached_chunks = json.load(f)
        app.logger.info("Loaded FAISS index and document chunks from disk.")
    else:
        app.logger.error("Index files not found. Please build the index first.")

# Call this function during app startup.
load_saved_index()

def get_embedding(text):
    try:
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=text
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        app.logger.error(f"Error getting embedding: {e}", exc_info=True)
        return None

def retrieve_relevant_chunks_faiss(query, top_k=2):
    if faiss_index is None or cached_chunks is None:
        return []
    query_emb = get_embedding(query)
    if query_emb is None:
        return []
    query_vec = np.array(query_emb, dtype="float32").reshape(1, -1)
    distances, indices = faiss_index.search(query_vec, top_k)
    return [cached_chunks[i] for i in indices[0] if i < len(cached_chunks)]

def rag_chat(query):
    if not cached_chunks or faiss_index is None:
        return "Error: Document index not initialized."
    
    relevant_chunks = retrieve_relevant_chunks_faiss(query, top_k=2)
    context = "\n\n".join(relevant_chunks)
    augmented_prompt = f"Context:\n{context}\n\nQuery: {query}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": augmented_prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        app.logger.error(f"Error in rag_chat: {e}", exc_info=True)
        return f"Error in rag_chat: {e}"

# ----- Load your models (they already exist in the models folder) -----
multi_tf_model = tf.keras.models.load_model(os.path.join('models', 'weightsfinal.h5'))
multi_xgb_model = joblib.load(os.path.join('models', 'xgb_clf_model_multi.pkl'))
binary_tf_model = tf.keras.models.load_model(os.path.join('models', 'weightb.h5'))
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
    img_stream = BytesIO(img_file.read())
    img_file.seek(0)
    img = keras_image.load_img(img_stream, target_size=(300, 300))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----- Load default details for OA and KL levels -----
DEFAULT_FILE_PATH = os.path.join("default.txt")
try:
    with open(DEFAULT_FILE_PATH, 'r') as f:
        default_details = f.read()
except Exception as e:
    default_details = "Default OA/KL level details not available."

# ----- Routes -----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clinical', methods=['GET', 'POST'])
def clinical():
    default_clinical_dict = dict(zip(clinical_columns, default_sample_data[0]))
    result_text = ""
    if request.method == 'POST':
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
        img_array = preprocess_image_file(file)
        predictions = multi_tf_model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=-1)[0]
        if predicted_class == 0:
            result_text = "X-ray Prediction: OA not detected (0)."
        else:
            result_text = f"X-ray Prediction: OA detected (1) with a KL score of {predicted_class}."
        return render_template('xray.html', result=result_text)
    return render_template('xray.html')

@app.route('/fusion', methods=['GET', 'POST'])
def fusion():
    default_clinical_dict = dict(zip(clinical_columns, default_sample_data[0]))
    result_text = ""
    if request.method == 'POST':
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
        if 'image_file' not in request.files:
            flash("No image file provided.")
            return redirect(request.url)
        file = request.files['image_file']
        if file.filename == '':
            flash("No file selected.")
            return redirect(request.url)
        img_array = preprocess_image_file(file)
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
        predictions_tf_binary = binary_tf_model.predict(img_array)
        xgb_probs_binary = binary_xgb_model.predict_proba(processed_df.values)
        w1_binary = 0.4897959183673469
        w2_binary = 0.5102040816326531
        fused_probs_binary = (w1_binary * predictions_tf_binary) + (w2_binary * xgb_probs_binary)
        final_label_binary = np.argmax(fused_probs_binary, axis=1)[0]
        binary_result = f"Binary Prediction: OA Detection = {'Yes (1)' if final_label_binary == 1 else 'No (0)'}."
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

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    response_text = ""
    user_message = ""
    if request.method == 'POST':
        user_message = request.form.get('message')
        if not user_message:
            app.logger.debug("No user message provided in POST request.")
            return jsonify({"response": "Please enter a message.", "user_message": ""})
        app.logger.debug(f"Received user message: {user_message}")
        try:
            # If you want to use RAG, comment out the simple chat below and uncomment the RAG call.
            # Simple Chat:
            # chat_response = openai.ChatCompletion.create(
            #     model="gpt-4o-mini",
            #     messages=[{"role": "user", "content": user_message}],
            #     temperature=0.7
            # )
            # response_text = chat_response.choices[0].message.content
            
            # RAG-based Chat:
            response_text = rag_chat(user_message)
            app.logger.debug(f"RAG answer: {response_text}")
        except Exception as e:
            app.logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
            response_text = f"Error calling OpenAI API: {e}"
    
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        app.logger.debug("AJAX request detected, returning JSON response.")
        return jsonify({"response": response_text, "user_message": user_message})
    
    app.logger.debug("Rendering chat.html template with response.")
    return render_template('chat.html', response=response_text, user_message=user_message)

if __name__ == '__main__':
    app.run(debug=True, port=8080)