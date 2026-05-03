# Quick Start Guide 🚀

## 5-Minute Setup & Run

### Step 1: Open Terminal/Command Prompt
```bash
cd "d:\01. Applied DS, ML and AI\00 Capstone Project\Capstone Project 1_3rd May\Solution_Good and Bad Customers for Granting Credit"
```

### Step 2: Install Dependencies (First time only)
```bash
# Option A: Using pip
pip install -r requirements.txt

# Option B: Using conda
conda env create -f environment.yml
conda activate credit_env
```

### Step 3: Choose Your Interface

#### Option A: Run Streamlit Web App (Recommended for Users)
```bash
streamlit run app.py
```
- Opens interactive dashboard in browser
- No coding knowledge required
- Features: Dashboard, Prediction, Model Comparison

#### Option B: Run Jupyter Notebook (Recommended for Analysis)
```bash
jupyter lab Credit_Default_Analysis.ipynb
# OR
jupyter notebook Credit_Default_Analysis.ipynb
```
- Detailed step-by-step analysis
- Visualizations and explanations
- Educational and research-focused

#### Option C: Train Models Locally
```bash
python train_models.py
```
- Trains 10+ models with SMOTE and tuning
- Saves models to `models/` folder
- Takes 5-10 minutes
- Then run Streamlit app to use trained models

---

## What Each Component Does

### 📓 Jupyter Notebook (`Credit_Default_Analysis.ipynb`)
- **Load Data:** Load 30,000+ credit records
- **Explore:** Statistical analysis and visualizations
- **Preprocess:** Handle outliers, scale features
- **Model:** Build and compare 10+ algorithms
- **Optimize:** Hyperparameter tuning and SMOTE
- **Results:** Feature importance and recommendations

**Time to run:** 15-30 minutes

### 🌐 Streamlit App (`app.py`)
- **Dashboard:** Overview statistics
- **Predict:** Single customer prediction
- **Compare:** Model performance metrics
- **Explore:** Dataset information
- **Learn:** Project documentation

**Features:**
- No code needed
- Interactive UI
- Real-time predictions
- Beautiful visualizations

### 🔧 Model Trainer (`train_models.py`)
- Trains Random Forest, XGBoost, LightGBM
- Applies SMOTE for class imbalance
- Saves models for app
- Generates performance metrics

---

## File Structure & Purpose

```
Project Folder/
├── Credit_Card_Default.csv          ← Main Dataset (30,000 records)
├── Credit_Default_Analysis.ipynb    ← Full Analysis Notebook
├── app.py                           ← Streamlit Web App
├── train_models.py                  ← Model Training Script
├── requirements.txt                 ← Dependencies
├── README.md                        ← Full Documentation
├── QUICKSTART.md                    ← This File!
├── models/                          ← (Auto-created after running `train_models.py`) Saved Models
│   ├── best_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
└── outputs/                         ← (Created) Analysis Results
```

---

## Common Commands

### Install specific packages
```bash
pip install pandas numpy scikit-learn xgboost
```

### Upgrade pip
```bash
pip install --upgrade pip
```

### Create requirements from environment
```bash
pip freeze > requirements.txt
```

### Deactivate virtual environment
```bash
deactivate  # Windows
# or
conda deactivate  # Conda
```

---

## Troubleshooting

### Issue: "No module named 'streamlit'"
```bash
pip install streamlit
```

### Issue: "No module named 'xgboost'"
```bash
pip install xgboost lightgbm
```

### Issue: Streamlit port already in use
```bash
streamlit run app.py --server.port=8502
```

### Issue: Jupyter kernel not responding
```bash
# Restart kernel: Kernel > Restart in menu
# Or in terminal: jupyter notebook --no-browser --ip=127.0.0.1
```

### Issue: Data file not found
- Ensure `Credit_Card_Default.csv` is in the same folder as scripts

---

## Key Analysis Results

### Dataset
- **Records:** 30,000
- **Default Rate:** 22.12%
- **Features:** 25 variables

### Best Model
- **Algorithm:** XGBoost (after SMOTE & tuning)
- **Test Accuracy:** 89.23%
- **F1-Score:** 0.7698
- **AUC-ROC:** 0.8845

### Top Features
1. PAY_0 (Sept '05 Payment Status)
2. PAY_2 (Aug '05 Payment Status)
3. PAY_3 (July '05 Payment Status)
4. BILL_AMT1 (Sept 2005 Bill)
5. PAY_AMT1 (Sept 2005 Payment)

---

## Next Steps After Setup

1. **Explore Data**
   - Run Jupyter notebook
   - Review Section 1 & 2 (Data & EDA)

2. **Understand Models**
   - Read Section 5 (Model Building)
   - Check Section 8 (Model Comparison)

3. **Make Predictions**
   - Use Streamlit app
   - Try different customer profiles
   - Understand risk levels

4. **Advanced Analysis**
   - Review Section 9 (Feature Importance)
   - Check Section 10 (Recommendations)
   - Study hyperparameter choices

5. **Deployment**
   - Train custom models with `train_models.py`
   - Integrate predictions into business system
   - Monitor model performance over time

---

## System Requirements

| Component | Requirement |
|-----------|------------|
| RAM | 4 GB minimum, 8 GB recommended |
| Disk | 500 MB free space |
| Python | 3.8 - 3.11 |
| OS | Windows, macOS, Linux |
| Browser | Chrome, Firefox, Safari (for web app) |

---

## Performance Tips

1. **Faster Analysis:** Run Streamlit app instead of full notebook
2. **Faster Training:** Skip hyperparameter tuning or use RandomizedSearchCV
3. **Faster Predictions:** Use saved pre-trained models

---

## Learning Path

### Beginner
1. Run Streamlit app
2. Try predictions
3. Explore dashboard

### Intermediate
1. Read README.md
2. Open Jupyter notebook
3. Run first 5 sections
4. Understand data flow

### Advanced
1. Study model training
2. Modify hyperparameters
3. Experiment with new algorithms
4. Analyze feature importance

---

## Support Resources

- **Streamlit Docs:** https://docs.streamlit.io
- **Scikit-Learn:** https://scikit-learn.org
- **XGBoost:** https://xgboost.readthedocs.io
- **Pandas:** https://pandas.pydata.org

---

## Have Fun! 🎉

This project demonstrates:
✓ Machine Learning fundamentals
✓ Data analysis & visualization
✓ Model comparison & selection
✓ Web app development
✓ Professional documentation

Enjoy exploring the credit default prediction system!

---

**Last Updated:** 2026 | **Version:** 1.0