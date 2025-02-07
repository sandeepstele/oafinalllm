# OA Detection and Fusion Model App with RAG-based Reporting

This repository contains a Flask-based web application for osteoarthritis (OA) prediction and detection. The application integrates multiple machine learning models to process clinical data and X-ray images, and it uses a Retrieval-Augmented Generation (RAG) pipeline to automatically generate patient reports. A separate FAISS index is used to quickly retrieve context from a document (e.g., `report_rules.txt`) to guide report generation.

## Features

- **Clinical Data Processing**
  - Accepts clinical parameters via a form.
  - Uses a pre-trained TensorFlow model and XGBoost classifiers for multi-class (including KL scoring) and binary OA predictions.
- **Image Processing**
  - Upload X-ray images for model prediction using TensorFlow-based image processing.
- **Fusion of Predictions**
  - Combines predictions from both clinical data and image analysis.
- **RAG-based Patient Report Generation**
  - Uses a separate FAISS index built from a report rules document (`report_rules.txt`) to retrieve relevant context.
  - Augments the patient report query with retrieved context and uses the OpenAI API (via a proxy) to generate a concise, evidence-based report.
- **Chatbot Integration**
  - Provides a chatbot interface (via the `/chat` route) that supports a RAG-based dialogue system for answering user queries.
- **Markdown Support**
  - A custom Jinja filter converts text wrapped in `**` to bold HTML, enhancing report readability.
- **Security & Configuration**
  - Uses environment variables (via python-dotenv) to securely manage API tokens and configuration settings.

## Prerequisites

- Python 3.8+
- pip
- Virtualenv (recommended)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/oafinalllm.git
    cd oafinalllm/oa_detection_app
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    A sample `requirements.txt` might include:

    ```
    Flask
    tensorflow
    numpy
    pandas
    joblib
    xgboost
    scikit-learn
    faiss-cpu
    openai
    python-dotenv
    ```

## Configuration

1. **Environment Variables:**

   Create a `.env` file in the root of the project (or in the `oa_detection_app` directory) with the following keys:

   ```init
   
   AIPROXY_TOKEN=your_actual_token_here
   API_BASE_URL=https://aiproxy.sanand.workers.dev/openai/v1 replace it with your base and token


   

   
