# 🩺 Diabetes Risk Assessment Predictor (Full-Stack SVM Network)

An enterprise-grade, full-stack machine learning application that predicts diabetes risk indicators in real-time. The system utilizes a **Support Vector Machine (SVM)** classifier trained on the benchmark **Pima Indians Diabetes Dataset** to map clinical consumer telemetry into risk probabilities, exposed via a high-performance **FastAPI** backend and an asynchronous, dynamic dark-themed dashboard.

---

## 📂 Core Project Architecture & Components

The workspace is highly optimized, containing exactly the essential assets required for production and validation:

| Component / File | Location | Description & System Responsibility |
| :--- | :--- | :--- |
| **`diabetes_dataset.csv`** | Root (`./`) | The official, real-world clinical data sourced from the National Institute of Diabetes and Digestive and Kidney Diseases. |
| **`models/scaler.pkl`** | `models/` | Serialized `StandardScaler` to normalize incoming user metrics against training set distributions. |
| **`models/svm_model.pkl`** | `models/` | Pre-trained **Support Vector Machine (SVM)** brain utilizing a non-linear **RBF Kernel** with internal probability calibration enabled. |
| **`server.py`** | Root (`./`) | **FastAPI Core Engine**. Manages the routing, processes the POST requests, computes automated data medians for missing indices, and executes the inference pipeline. |
| **`index.html`** | `frontend/` | **Web UI Dashboard**. An asynchronous interface utilizing JavaScript `fetch()` architectures to handle dynamic, color-coded state transitions based on threshold alerts. |

---

## 🧠 Machine Learning Methodology

The core intelligence layer relies heavily on standard computer science data frameworks:
1. **High-Dimensional Hyperplane Slicing:** The SVM interprets metrics (Glucose, BMI, Age) as coordinates within an abstract feature hyper-space, evaluating which side of the trained 29-dimensional decision boundary wall (Hyperplane) the transaction vector sits on.
2. **The Radial Basis Function (RBF) Kernel:** To solve the non-linear distributions inherent to clinical diagnostic patterns, an RBF kernel mathematical trick maps the vectors into higher dimensional spaces to draw a curved defensive perimeter separating high-risk profiles from healthy baselines.
3. **Platt Scaling Calibration:** Since basic SVMs only output binary decisions (0 or 1), Platt Scaling maps the geometric distance margins of vectors from the hyperplane into clean, readable floating-point probability percentages for the user interface.

---

## 🚀 Step-by-Step Deployment Guide

Follow these sequential terminal commands to launch the full-stack system locally:

### 1. Initialize the FastAPI Gateway
Open your terminal window in the project root directory where `server.py` resides and execute:
```bash
python server.py
```
The terminal will initialize a local server link, monitoring incoming requests at `http://127.0.0.1:8000`.

### 2. Launch the Client Frontend
* **Method A (Live Server):** Open `frontend/index.html` inside VS Code, right-click, and select "Open with Live Server".
* **Method B (Native Browser):** Navigate to the `frontend/` folder within your operating system's file explorer and double-click `index.html`.

### 3. Clear Environment Cache
Once the web app loads, force a total browser hard-refresh to clear old template frames:
* **Windows / Linux:** `Ctrl` + `F5`
* **Mac:** `Cmd` + `Shift` + `R`

---

## 🧪 Production Validation Profiles
Use these three calibrated archetypes to demonstrate active model performance during evaluations:

* 🟢 **Baseline Optimal Profile:** Glucose: 85 | BMI: 21.5 | Age: 24 ──► Result: **✅ OPTIMAL HEALTH INDICATORS** (Low Risk)
* 🟡 **Borderline Warning Profile:** Glucose: 135 | BMI: 28.0 | Age: 38 ──► Result: **⚠️ MODERATE RISK ENVELOPE** (Warning)
* 🔴 **Critical Anomaly Profile:** Glucose: 185 | BMI: 39.5 | Age: 52 ──► Result: **🚨 HIGH RISK PROFILE DETECTED** (Crimson Alert)

---

*Developed as an academic defense project aligning full-stack software development patterns with machine learning engineering paradigms.*
