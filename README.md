# 🧠 FaceBoard
### Hands-free Virtual Keyboard and Mouse using Facial Tracking

FaceBoard is a Python-based system that lets you **control your keyboard and mouse using only your face**.  
It uses your webcam to track facial landmarks (via [MediaPipe Face Mesh](https://developers.google.com/mediapipe)) and translates movements into actions.

---

## 🎯 Features

✅ Face-controlled **cursor movement**  
✅ Mouth-gesture-based **clicking and typing**  
✅ On-screen **virtual keyboard**  
✅ Supports **Shift**, **Caps**, **Symbols**, and **mouse control (D-Pad)**  
✅ Real-time tracking with **OpenCV** and **MediaPipe**

---

## 🧩 Tech Stack

- **Python 3.8+**
- **OpenCV** — for webcam and drawing interface  
- **MediaPipe** — for facial landmark tracking  
- **PyAutoGUI** — for controlling the OS keyboard/mouse  

---

## 🧰 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/eggeol/FaceBoard.git
   cd FaceBoard

2. **Create a virtual environment (optional but recommended)**

   ```bash
   Copy code
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows

3. **Install dependencies**

   ```bash
   Copy code
   pip install -r requirements.txt

4. **Run the program**

   ```bash
   Copy code
   python faceboard.py
