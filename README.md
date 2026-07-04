# 🏥 Healthcare AI Models

A personal research repository of Machine Learning, Deep Learning, and Reinforcement Learning models developed for healthcare and medical analysis — organized by disease and medical condition.

> *This repository is a living archive of my ongoing research in applied AI for healthcare and medicine.*

---

## 📌 Overview

Each folder in this repository represents a specific disease or medical condition, and contains all models, experiments, notebooks, and results related to that condition. Models range from classical ML to deep neural networks and RL agents, chosen based on what best suits each problem.

---

## 🗂️ Repository Structure

```
healthcare-models/
│
├── huntingtons-disease/
│   ├── data/                  # Dataset references & preprocessing
│   ├── models/                # Trained models & checkpoints
│   ├── notebooks/             # Experiments & analysis
│   ├── results/               # Evaluation metrics & plots
│   └── README.md              # Disease-specific documentation
│
├── heart-disease/
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   ├── results/
│   └── README.md
│
├── oral-cancer/
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   ├── results/
│   └── README.md
│
├── [disease-name]/            # Each new disease follows the same structure
│   └── ...
│
├── utils/                     # Shared utilities across all disease modules
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── visualization.py
│
├── requirements.txt
└── README.md
```

---

## 🧬 Diseases & Conditions

| # | Disease / Condition | ML Approach | Data Type | Status |
|---|---------------------|-------------|-----------|--------|
| 1 | Huntington's Disease | — | — | 🔄 In Progress |
| 2 | Heart Disease | — | — | 🔄 In Progress |
| 3 | Oral Cancer | — | — | 🔄 In Progress |

> This table will be updated as models are developed and results are available.

---

## 📁 Each Disease Folder Contains

- **`data/`** — dataset links, data cards, and preprocessing scripts
- **`models/`** — model architectures, training scripts, and saved checkpoints
- **`notebooks/`** — exploratory data analysis and experiment notebooks
- **`results/`** — evaluation metrics, plots, and comparison against baselines
- **`README.md`** — methodology, dataset details, and findings for that condition

---

## ⚙️ Setup & Usage

### Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended for deep learning models)

### Installation

```bash
git clone https://github.com/your-username/healthcare-models.git
cd healthcare-models
pip install -r requirements.txt
```

### Running a Model

Navigate into any disease folder and follow its own `README.md`:

```bash
cd heart-disease
python models/train.py --config config.yaml
python models/evaluate.py --model_path checkpoints/best_model.pth
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Frameworks | PyTorch, TensorFlow / Keras |
| ML Libraries | scikit-learn, XGBoost, LightGBM |
| Medical Imaging | MONAI, OpenCV, SimpleITK, pydicom |
| Data & Analysis | NumPy, Pandas, Matplotlib, Seaborn |
| Experiment Tracking | Weights & Biases, MLflow |
| RL Libraries | Stable-Baselines3, RLlib |

---

## ⚠️ Disclaimer

The models and findings in this repository are developed **strictly for research and educational purposes**. They have **not** been clinically validated and are **not** intended for real-world medical diagnosis, treatment planning, or patient care decisions. Always consult qualified medical professionals for clinical matters.

---

## 📬 Contact

For questions or research discussions, feel free to reach out via GitHub Issues.

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, reference, or build upon this work for research purposes. Please cite or credit this repository where applicable.

See the [LICENSE](LICENSE) file for full details.
