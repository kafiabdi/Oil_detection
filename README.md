# Oil Spill Detection System

## Overview
The **Oil Spill Detection System** is an AI-powered solution designed to detect and segment oil spills in images using deep learning. The system leverages convolutional neural networks (CNNs) and custom segmentation models to analyze images and generate precise predictions of oil spill regions.

---

## Features
- Detects oil spill areas in images with high accuracy.
- Supports multiple input images for batch analysis.
- Outputs segmented images highlighting affected regions.
- Logs detections and predictions for analysis (`alert_log.txt` and `inspection_log.csv`).
- Modular code structure for easy model updates and experimentation.

---

## Project Structure

```

Oil_detection/
│
├── app.py                       # Main application script
├── oil_spill_detection.py       # Detection logic and model inference
├── option2.py                   # Alternative processing pipeline
├── requirements.txt             # Required Python packages
├── pole_segmentation_model.keras# Trained segmentation model
├── unet_final.h5 / .keras       # Trained UNet model files
├── unet_savedmodel.zip          # Saved TensorFlow model
├── LADOS.v2i.png-mask-semantic/ # Dataset folder (ignored in git)
├── venv/                        # Virtual environment (ignored in git)
└── README.md                     # Project documentation

````

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/kafiabdi/Oil_detection.git
````

2. Navigate to the project directory:

```bash
cd Oil_detection
```

3. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

4. Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main application:

```bash
python app.py
```

* The app processes images from the input folder.
* Predicted outputs and logs are saved in the `runs/` folder.
* Supports batch processing for multiple images.

---

## Model Details

* **UNet** model for segmentation of oil spill regions.
* Trained on labeled datasets from `LADOS.v2i.png-mask-semantic/`.
* Saved models are included (`.h5`, `.keras`, `.zip`) for inference.

---

## Notes

* Large model files are included, ensure your system has enough memory for processing.
* Virtual environment (`venv/`) and dataset folder (`LADOS.v2i.png-mask-semantic/`) are ignored in Git.

---

## Author

**Kafi Abdi**

* GitHub: [kafiabdi](https://github.com/kafiabdi)
* Email: [abdik9927@gmail.com](mailto:abdik9927@gmail.com)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
