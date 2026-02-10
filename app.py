from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import os
import numpy as np
import random
import base64
from datetime import datetime

app = Flask(__name__)
app.secret_key = "super_secret_exam_key"

# Path calculations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEHAVIOR_MODEL_PATH = os.path.join(BASE_DIR, "ml", "behavior_model.pkl")
PERFORMANCE_MODEL_PATH = os.path.join(BASE_DIR, "ml", "performance_model.pkl")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "static", "snapshots")

if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

# Master Question Bank
QUESTION_BANK = [
    {"id": "q1", "q": "What is the primary goal of Supervised Learning?", "a": "Clustering data", "b": "Predicting labels based on input", "c": "Data compression", "d": "Finding hidden patterns", "ans": "b"},
    {"id": "q2", "q": "Which algorithm is best suited for binary classification?", "a": "Linear Regression", "b": "K-Means", "c": "Logistic Regression", "d": "PCA", "ans": "c"},
    {"id": "q3", "q": "What does 'K' represent in K-Nearest Neighbors?", "a": "Number of clusters", "b": "Number of neighboring points", "c": "Kernel function", "d": "Knowledge base", "ans": "b"},
    {"id": "q4", "q": "In Deep Learning, what is an 'Epoch'?", "a": "One pass of the entire dataset", "b": "A single training instance", "c": "The number of layers", "d": "The learning rate", "ans": "a"},
    {"id": "q5", "q": "Which library is primarily used for numerical operations in Python?", "a": "Pandas", "b": "NumPy", "c": "Matplotlib", "d": "Flask", "ans": "b"},
    {"id": "q6", "q": "What is 'Overfitting'?", "a": "Model performs well on test but poor on train", "b": "Model performs poor on both", "c": "Model performs well on train but poor on test", "d": "Model is extremely accurate", "ans": "c"},
    {"id": "q7", "q": "Which activation function is most common in hidden layers of a Neural Network?", "a": "Sigmoid", "b": "ReLU", "c": "Softmax", "d": "Tanh", "ans": "b"},
    {"id": "q8", "q": "What is the full form of CNN in ML?", "a": "Computer Network Node", "b": "Convolutional Neural Network", "c": "Centralized Neural Node", "d": "Continuous Network Node", "ans": "b"},
    {"id": "q9", "q": "Which of these is an Unsupervised Learning task?", "a": "Regression", "b": "Classification", "c": "Clustering", "d": "Reinforcement", "ans": "c"},
    {"id": "q10", "q": "What is the 'Bias' in a Machine Learning model?", "a": "Random noise", "b": "Error from overly simplistic assumptions", "c": "Error from high sensitivity to noise", "d": "Initial weights", "ans": "b"},
    {"id": "q11", "q": "Who is considered the father of Artificial Intelligence?", "a": "Alan Turing", "b": "John McCarthy", "c": "Elon Musk", "d": "Andrew Ng", "ans": "b"},
    {"id": "q12", "q": "In Python, what keyword is used to define a function?", "a": "func", "b": "def", "c": "lambda", "d": "define", "ans": "b"},
    {"id": "q13", "q": "Which data structure uses LIFO (Last In First Out)?", "a": "Queue", "b": "Stack", "c": "Linked List", "d": "Tree", "ans": "b"},
    {"id": "q14", "q": "What is the main purpose of PCA?", "a": "Dimensionality Reduction", "b": "Error Correction", "c": "Data Cleaning", "d": "Clustering", "ans": "a"},
    {"id": "q15", "q": "Which ML technique is used for AlphaGo?", "a": "Supervised Learning", "b": "Unsupervised Learning", "c": "Reinforcement Learning", "d": "Semi-supervised", "ans": "c"},
    {"id": "q16", "q": "What does 'SVM' stand for?", "a": "Simple Vector Machine", "b": "Support Vector Machine", "c": "System Value Monitor", "d": "Scaled Vector Mode", "ans": "b"},
    {"id": "q17", "q": "Which of the following is NOT a type of Neural Network?", "a": "RNN", "b": "ANN", "c": "CNN", "d": "DBA", "ans": "d"},
    {"id": "q18", "q": "What is the purpose of a cross-validation?", "a": "To speed up training", "b": "To evaluate model performance more reliably", "c": "To clean the dataset", "d": "To reduce data size", "ans": "b"},
    {"id": "q19", "q": "What is a 'Tensor' in TensorFlow?", "a": "A multi-dimensional array", "b": "A neural layer", "c": "A GPU component", "d": "A logic gate", "ans": "a"},
    {"id": "q20", "q": "Which metric is most suitable for evaluating a regression model?", "a": "Accuracy", "b": "F1-Score", "c": "MSE", "d": "Precision", "ans": "c"},
    {"id": "q21", "q": "In Random Forest, what are the basic building blocks?", "a": "Neurons", "b": "Decision Trees", "c": "Clusters", "d": "Support Vectors", "ans": "b"},
    {"id": "q22", "q": "What is 'feature engineering'?", "a": "Building hardware for AI", "b": "Selecting/Transforming data for ML", "c": "Cleaning databases", "d": "Adding GPU memory", "ans": "b"},
    {"id": "q23", "q": "Which Python library is famous for Data Visualization?", "a": "Seaborn", "b": "PyPlot", "c": "Plotly", "d": "All of the above", "ans": "d"},
    {"id": "q24", "q": "What does NLP stand for?", "a": "Natural Level Process", "b": "Neutral Link Program", "c": "Natural Language Processing", "d": "Node Link Protocol", "ans": "c"},
    {"id": "q25", "q": "Which of these is a popular NLP model?", "a": "BERT", "b": "K-Means", "c": "Random Forest", "d": "Logistic Regression", "ans": "a"},
    {"id": "q26", "q": "What is the 'Softmax' function used for?", "a": "Regression output", "b": "Probability distribution in multi-class", "c": "Input normalization", "d": "Feature scaling", "ans": "b"},
    {"id": "q27", "q": "What is 'Gradient Descent'?", "a": "An optimization algorithm", "b": "A type of neural network", "c": "A data cleaning tool", "d": "A hardware component", "ans": "a"},
    {"id": "q28", "q": "Which company developed TensorFlow?", "a": "Facebook", "b": "Microsoft", "c": "Google", "d": "Amazon", "ans": "c"},
    {"id": "q29", "q": "What is 'Transfer Learning'?", "a": "Moving data to cloud", "b": "Using pre-trained model for new task", "c": "Training on multiple CPUs", "d": "Sending models to users", "ans": "b"},
    {"id": "q30", "q": "In ML, what is 'Standardization'?", "a": "Cleaning text data", "b": "Scaling features to zero mean/unit variance", "c": "Removing outliers", "d": "Label encoding", "ans": "b"}
]

# Real-time Monitoring Storage
# Real-time Monitoring Storage
ACTIVE_EXAMS = {}
PAST_RESULTS = []
LOGIN_LOGS = []

# Admin Credentials
ADMIN_CONFIG = {
    "email": "admin@exam.com",
    "password": "admin123"
}

# Load ML Models
try:
    behavior_model = joblib.load(BEHAVIOR_MODEL_PATH)
    performance_model = joblib.load(PERFORMANCE_MODEL_PATH)
except Exception as e:
    print(f"Error loading models: {e}")
    behavior_model = None
    performance_model = None

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get('email')
        session['user'] = email
        login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session['login_time'] = login_time
        
        # Log the login
        LOGIN_LOGS.append({
            'email': email,
            'time': login_time,
            'ip': request.remote_addr
        })
        
        return redirect(url_for('dashboard'))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("dashboard.html")

@app.route("/exam")
def exam():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    shuffled_questions = random.sample(QUESTION_BANK, 15)
    session['current_exam_questions'] = [q['id'] for q in shuffled_questions]
    
    user_email = session['user']
    ACTIVE_EXAMS[user_email] = {
        'status': 'Active',
        'face_status': 'Initializing...', # New field
        'tab_switches': 0,
        'face_violations': 0,
        'looking_away_count': 0,
        'login_time': session.get('login_time'),
        'admin_msg': None,
        'terminate_signal': False,
        'snapshot_request': False,
        'last_snapshot': None,
        'is_streaming': False,
        'messages': [] # Chat history
    }
    
    return render_template("exam.html", questions=shuffled_questions)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    user_email = session.get('user')
    if not user_email or user_email not in ACTIVE_EXAMS:
        return jsonify({"status": "error"}), 403
    
    msg = request.json.get('message')
    if msg:
        ACTIVE_EXAMS[user_email]['messages'].append({
            'sender': 'Student',
            'text': msg,
            'time': datetime.now().strftime("%H:%M")
        })
    return jsonify({"status": "success"})



@app.route("/api/telemetry", methods=["POST"])
def telemetry():
    user_email = session.get('user')
    if not user_email or user_email not in ACTIVE_EXAMS:
        return jsonify({"status": "error"}), 403
    
    data = request.json
    ACTIVE_EXAMS[user_email].update({
        'tab_switches': data.get('tab_switches', 0),
        'face_violations': data.get('face_violations', 0),
        'looking_away_count': data.get('looking_away_count', 0),
        'face_status': data.get('face_status', 'Unknown') # Update face status
    })
    
    resp = {
        "admin_msg": ACTIVE_EXAMS[user_email]['admin_msg'],
        "terminate": ACTIVE_EXAMS[user_email]['terminate_signal'],
        "snapshot_request": ACTIVE_EXAMS[user_email]['snapshot_request'],
        "is_streaming": ACTIVE_EXAMS[user_email]['is_streaming'],
        "messages": ACTIVE_EXAMS[user_email]['messages'] # Sync chat
    }
    
    # Reset messages once sent? No, keep full history for now or rely on client to dedupe.
    # Ideally, client tracks last message index. For simplicity, sending full history.
    ACTIVE_EXAMS[user_email]['admin_msg'] = None
    ACTIVE_EXAMS[user_email]['snapshot_request'] = False
    
    return jsonify(resp)

@app.route("/api/upload-snapshot", methods=["POST"])
def upload_snapshot():
    user_email = session.get('user')
    if not user_email or user_email not in ACTIVE_EXAMS:
        return jsonify({"status": "error"}), 403
    
    data = request.json
    img_data = data.get('image')
    if img_data:
        # Save image (overwrite last_snapshot to save space)
        filename = f"{user_email.replace('@', '_').replace('.', '_')}_live.jpg"
        filepath = os.path.join(SNAPSHOT_DIR, filename)
        
        with open(filepath, "wb") as fh:
            fh.write(base64.b64decode(img_data.split(",")[1]))
            
        ACTIVE_EXAMS[user_email]['last_snapshot'] = filename
        return jsonify({"status": "success", "filename": filename})
    
    return jsonify({"status": "failed"}), 400

# --- ADMIN ROUTES ---

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        if email == ADMIN_CONFIG['email'] and password == ADMIN_CONFIG['password']:
            session['admin'] = True
            return redirect(url_for('admin'))
        return "Invalid Credentials"
    return render_template("admin_login.html")

@app.route("/admin/logout")
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('admin_login'))

@app.route("/admin")
def admin():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    return render_template("admin.html", active_exams=ACTIVE_EXAMS, results=PAST_RESULTS, logs=LOGIN_LOGS)

@app.route("/admin/warn", methods=["POST"])
def admin_warn():
    if not session.get('admin'): return jsonify({"status": "unauthorized"}), 401
    email = request.json.get('email')
    msg = request.json.get('msg')
    if email in ACTIVE_EXAMS:
        ACTIVE_EXAMS[email]['admin_msg'] = msg
        return jsonify({"status": "success"})


@app.route("/admin/terminate", methods=["POST"])
def admin_terminate():
    if not session.get('admin'): return jsonify({"status": "unauthorized"}), 401
    email = request.json.get('email')
    if email in ACTIVE_EXAMS:
        ACTIVE_EXAMS[email]['terminate_signal'] = True
        return jsonify({"status": "success"})
    return jsonify({"status": "user_not_found"}), 404

@app.route("/admin/request-snapshot", methods=["POST"])
def admin_request_snapshot():
    if not session.get('admin'): return jsonify({"status": "unauthorized"}), 401
    email = request.json.get('email')
    if email in ACTIVE_EXAMS:
        ACTIVE_EXAMS[email]['snapshot_request'] = True
        return jsonify({"status": "success"})
    return jsonify({"status": "user_not_found"}), 404

@app.route("/admin/toggle-stream", methods=["POST"])
def admin_toggle_stream():
    if not session.get('admin'): return jsonify({"status": "unauthorized"}), 401
    email = request.json.get('email')
    stream_state = request.json.get('stream') # boolean
    if email in ACTIVE_EXAMS:
        ACTIVE_EXAMS[email]['is_streaming'] = stream_state
        return jsonify({"status": "success", "streaming": stream_state})
    return jsonify({"status": "user_not_found"}), 404

@app.route("/submit", methods=["POST"])
def submit():
    if 'user' not in session:
        return redirect(url_for('login'))

    user_email = session['user']
    is_terminated = request.form.get('terminated') == 'true'
    submit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    score = 0
    total_q = 0
    if 'current_exam_questions' in session:
        total_q = len(session['current_exam_questions'])
        for q_id in session['current_exam_questions']:
            user_ans = request.form.get(q_id)
            correct_ans = next((q['ans'] for q in QUESTION_BANK if q['id'] == q_id), None)
            if user_ans == correct_ans:
                score += 1

    time_spent = int(request.form.get('time_spent', 0))
    answer_changes = int(request.form.get('answer_changes', 0))
    tab_switches = int(request.form.get('tab_switches', 0))
    face_violations = int(request.form.get('face_violations', 0))
    looking_away_count = int(request.form.get('looking_away_count', 0))
    
    start_time = datetime.strptime(session['login_time'], "%Y-%m-%d %H:%M:%S")
    end_time_dt = datetime.strptime(submit_time, "%Y-%m-%d %H:%M:%S")
    duration_minutes = (end_time_dt - start_time).total_seconds() / 60

    risk_level = "Low"
    suspicion_score = 0
    
    if is_terminated:
        risk_level = "TERMINATED (FRAUD)"
        suspicion_score = 100
    elif behavior_model:
        # Isolation Forest prediction
        behavior_data = np.array([[time_spent, answer_changes, tab_switches]])
        pred = behavior_model.predict(behavior_data)[0] # 1 for inlier, -1 for outlier
        score_val = behavior_model.decision_function(behavior_data)[0] # Negative is anomalous
        
        # Normalize decision function to 0-100 suspicion score
        # decision_function yields positive for normal, negative for abnormal. 
        # Typical range might be -0.2 to 0.2. Let's invert and scale.
        # We want negative numbers (outliers) to have HIGH suspicion.
        suspicion_score = max(0, min(100, (0.5 - score_val) * 100)) # Simple heuristic scaling
        
        if pred == -1 or tab_switches > 3 or face_violations > 5:
             risk_level = "High"
        elif suspicion_score > 40 or tab_switches > 1:
             risk_level = "Medium"

    predicted_performance = 0
    if performance_model and not is_terminated:
        scaled_score = (score / total_q) * 10 if total_q > 0 else 0
        score_data = np.array([[scaled_score]])
        predicted_performance = round(performance_model.predict(score_data)[0], 2)

    final_score_val = (score / total_q * 100) if total_q > 0 else 0
    PAST_RESULTS.append({
        'email': user_email,
        'score': score,
        'total': total_q,
        'percentage': final_score_val,
        'risk': risk_level,
        'suspicion': suspicion_score,
        'login_time': session.get('login_time'),
        'submit_time': submit_time,
        'duration': round(duration_minutes, 1)
    })

    if user_email in ACTIVE_EXAMS:
        del ACTIVE_EXAMS[user_email]

    # Calculate Class Average
    all_scores = [r['percentage'] for r in PAST_RESULTS]
    class_avg = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0

    return render_template(
        "result.html",
        score=f"{score}/{total_q}" if not is_terminated else "0 (Terminated)",
        risk=risk_level,
        prediction=predicted_performance if not is_terminated else "N/A",
        terminated=is_terminated,
        user_percentage=final_score_val,
        class_average=class_avg
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
