# 🎙️ Audio Emotion Detection

A deep learning-powered system to detect human emotions from speech audio clips. This project achieves **~95% accuracy** and successfully identifies emotions like *happy*, *sad*, *angry*, *fear*, and more using MFCC features and an optimized CNN model.

---

## 🚀 Highlights

- 🎧 Audio input: `.wav` files
- 🧠 Model: 1D Convolutional Neural Network with BatchNorm and Dropout
- 🎚️ Feature extraction: MFCC (mean + std)
- 🔁 Augmentation: Noise, pitch shift, time stretch, EQ, masking
- ✅ Validation Accuracy: **~95%**
- 🔍 Example Test: Correctly predicted **"fear"** with **97% confidence**

---

## 📁 Project Structure

```
Audio_Emotion_Detection/
│
├── models/                 # Saved model, class mapping, accuracy plots
├── dataset/                # Input audio files
├── src/                    # Source code
│   ├── train.py            # Model training
│   ├── test.py             # Emotion prediction
│   ├── extract_features.py # MFCC extraction
│   ├── data_loader.py      # Dataset loader
│   └── augmentation.py     # Data augmentation methods
├── features.csv            # Extracted MFCC features
├── requirements.txt        # Project dependencies
└── README.md
```

---

## 🛠️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Ashfaq-Hussain7/Audio_Emotion_Detection.git
cd Audio_Emotion_Detection
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

---

## 🎓 Training

### Step 1: Extract Features
```bash
python src/extract_features.py
```

### Step 2: Train the Model
```bash
python src/train.py
```

Training includes:
- 5-fold Stratified Cross-Validation
- Early stopping & class-weighted loss
- Augmentation-enabled feature extraction

---

## 🧪 Testing

### Step 1: Run Inference
```bash
python src/test.py
```

Set your audio path in `test.py`:
```python
predict_emotion("audio_samples/fear_test.wav")
```

### ✅ Sample Output:
```
Predicted Emotion: fear
Confidence: 97.02%
```

---

## 📊 Performance

- ✅ **Validation Accuracy**: ~95% (5-fold CV)
- 🧪 Tested on real unseen audio: 97% confidence on "fear"
- 📈 Training and validation curves saved in `models/training_curves.png`

---

## 📦 Future Work

- Integrate mel-spectrogram and spectrogram feature options
- Add a CNN-LSTM variant for sequential modeling
- Deploy real-time detection via Streamlit or Flask
- Support for more nuanced emotions (boredom, sarcasm, etc.)

---

## 📂 Dataset Sources

You can use any `.wav` file, but for best results, consider:
- [RAVDESS](https://zenodo.org/record/1188976)
- [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

---

## 🙌 Author

**Ashfaq Hussain**  
[GitHub: @Ashfaq-Hussain7](https://github.com/Ashfaq-Hussain7)

---

