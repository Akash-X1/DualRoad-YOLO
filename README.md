# Optimized Day and Low-Light Road Damage Detection with Automated Risk Assessment

## README

This folder contains the final MSc project code and trained models.

### Project folder

```text
Optimized Day and Low-Light Road Damage Detection with Automated Risk/
│
├── templates/
│   ├── dual_index.html
│   └── index.html
│
├── pothole weights/
│   ├── best.pt
│   ├── last.pt
│   └── best.onnx
│
├── crack weights/
│   ├── best.pt
│   ├── last.pt
│   └── best.onnx
│
├── app.py
├── dual_detect_app.py
├── dual_live_detect.py
├── final4.ipynb
├── crack3.ipynb
├── README.md
└── requirements.txt
```

`dual_index.html` is used by the final dual-model web application.  
`index.html` is used by the earlier model-selection application.

---

## 1. Install Requirements

Open a terminal inside the project folder and run:

```bat
pip install -r requirements.txt
```

---

## 2. Paths to Change

### `dual_detect_app.py`

**Pothole model path**

Change:

```python
POTHOLE_MODEL_PATH = ...
```

to the location of:

```text
pothole weights\best.pt
```

**Crack model path**

Change:

```python
CRACK_MODEL_PATH = ...
```

to the location of:

```text
crack weights\best.pt
```

---

### `dual_live_detect.py`

**Pothole model path**

Change:

```python
POTHOLE_MODEL_PATH = ...
```

to the location of:

```text
pothole weights\best.pt
```

**Crack model path**

Change:

```python
CRACK_MODEL_PATH = ...
```

to the location of:

```text
crack weights\best.pt
```

---

### `final4.ipynb`

**Cell 2 — Pothole dataset path**

Change the dataset path to the location of the Final 4 pothole datasets.

**Cell 9 — Pothole model path**

Change the model path to:

```text
pothole weights\best.pt
```

---

### `crack3.ipynb`

**Cell 2 — Crack dataset paths**

Change the Japan and US crack dataset paths to their correct locations.

**Cell 10 — Crack model path**

Change the model path to:

```text
crack weights\best.pt
```

**Cell 10 — Dataset YAML path**

If validation is being run, change the YAML path to the correct Crack 3 dataset YAML file.

---

## 3. Run the Final Web Application

Run:

```bat
python dual_detect_app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

Upload an image or video. The pothole and crack results will be shown separately.

Press `Ctrl + C` in the terminal to stop the server.

---

## 4. Run Live Detection

Run:

```bat
python dual_live_detect.py
```

Two windows will open:

- Pothole Detection — Final 4
- Crack Detection — Crack 3

Press `q` to stop.
