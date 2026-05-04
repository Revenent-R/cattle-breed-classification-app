# 🐄 Cattle Breed Classifier

An AI-powered application for **real-time and image-based cattle detection and breed classification** using YOLO, with automatic focus on the **closest cattle in view**.

---

## ✨ Key Highlights

* Detects multiple cattle and selects the **closest one (largest bounding box)**
* Smooth zoom-in transition to focus on selected cattle
* Supports **both real-time webcam and image input**
* Multi-language cattle information display
* Confidence-based visual feedback system

---

## 🚀 Features

* 📸 **Image-based classification**

  * Upload an image and detect cattle
  * Automatically zoom into closest cattle
  * Predict breed with confidence score

* 🎥 **Real-time detection**

  * Live webcam feed (local execution only)
  * Detects cattle continuously
  * Capture and classify selected frame
  * ⚠️ Not available in deployed version (cloud platforms do not support OpenCV webcam access)

* 🎯 **Closest cattle focus**

  * Uses bounding box area to determine proximity
  * Selects and zooms into nearest cattle

* 🌐 **Multilingual support**

  * English, Hindi, Marathi, Tamil, Telugu, Malayalam, Kannada
  * Displays detailed breed information

---

## 🧠 Core Logic

1. Detect cattle using YOLO object detection
2. Identify all cattle in the frame
3. Compute bounding box areas
4. Select the **largest bounding box (closest cattle)**
5. Apply progressive cropping + zoom animation
6. Classify breed using trained model
7. Display confidence + detailed breed info

---

## 🛠 Tech Stack

* Python
* Streamlit
* YOLO (Ultralytics)
* OpenCV
* PyTorch
* NumPy
* PIL

---

## 📦 Models Used

* `best_19.pt` → Breed classification model
* `best_latest_detect.pt` → Cattle detection model

---

## 📂 Project Structure

```
cattle-breed-classifier/
│── app.py
│── models/
│    ├── best_19.pt
│    └── best_latest_detect.pt
│── data/
│    └── cattle_info_updated.json
│── requirements.txt
│── README.md
```

---

## 🐄 Supported Breeds

- Alambadi  
- Amritmahal  
- Ayrshire  
- Banni  
- Bargur  
- Bhadawari  
- Brown Swiss  
- Dangi  
- Deoni  
- Gir  
- Hariana  
- Holstein Friesian  
- Jersey  
- Nagpuri  
- Rathi  
- Red Sindhi  
- Sahiwal  
- Tharparkar  

---

## ⚙️ Installation & Run

### 1. Clone the repository

```bash
git clone https://github.com/Revenent-R/cattle-breed-classification-app
cd cattle-breed-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---


## 🌐 Live Demo

https://cattle-breed-classification-app.streamlit.app/

⚠️ **Deployment Note**

- The live demo supports:
  - Image upload classification ✅
  - Closest cattle detection & zoom ✅
  - Multilingual information display ✅  

- The following feature is **disabled in the deployed version**:
  - ❌ Real-time webcam detection (OpenCV `cv2.VideoCapture` not supported on cloud platforms)

👉 To use real-time mode, run the app locally on your system.

---

## 🧪 Example Workflow

* Upload an image containing multiple cattle
* System detects all cattle
* Selects the closest one
* Applies zoom and enhancement
* Outputs:

  * Predicted breed
  * Confidence score
  * Detailed breed information

---

## 🌍 Multilingual Support

Supports breed information in:

* English
* Hindi
* Marathi
* Tamil
* Telugu
* Malayalam
* Kannada

---