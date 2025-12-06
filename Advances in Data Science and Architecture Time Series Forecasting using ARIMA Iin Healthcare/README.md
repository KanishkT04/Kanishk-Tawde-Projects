# Time Series Forecasting with ARIMA in Healthcare  
### Forecasting Daily Emergency Department (ED) Patient Volumes

INFO 7390 – Advanced Data Science and Architecture  
Author: **Kanishk Tawde**  

# 1. Concept Overview

Hospitals face significant uncertainty in daily Emergency Department (ED) patient volumes.  
Accurate short-term forecasting helps with:

- Nurse and physician staffing
- Bed allocation and capacity planning
- Reducing overcrowding and wait times
- Managing operational flow during seasonal surges (e.g., flu, RSV)

This repository teaches **Time Series Forecasting using the ARIMA model** through an  
industry-relevant healthcare example. It is designed as part of the  
**INFO 7390 Take-Home Final: Teaching Data Science Concepts**.

Learners will gain both conceptual understanding and practical modeling skills, including:

- Identifying trends, seasonality, and noise in time series data  
- Understanding stationarity and applying the Augmented Dickey–Fuller (ADF) test  
- Using ACF & PACF plots to select ARIMA(p, d, q) parameters  
- Fitting ARIMA models using `statsmodels`  
- Generating forecasts with confidence intervals  
- Evaluating models using MAE and RMSE  
- Interpreting forecasts from a **hospital operations perspective**

---

# 2. Repository Structure

time-series-arima-healthcare/
├─ notebooks/
│   ├─ KanishkTawde_TimeSeriesForecastingARIMAHealthcare_WorkingCodeImplementation     # Instructor version: full, commented pipeline
│   └─ KanishkTawde_TimeSeriesForecastingARIMAHealthcare_StarterTemplate    # Student template with TODOs
│
├─ data/
│   └─ daily_ed_visits.csv                    # Synthetic ED dataset (3 years)
│
├─ docs/                                      # Tutorial PDF, figures, pedagogical report, assessment materials
│
├─ slides/                                    # Show-and-Tell presentation materials
│
├─ requirements.txt
└─ README.md


# 3. Installation Instructions

Step 1 — Clone the Repository
git clone https://github.com/<your-username>/time-series-arima-healthcare.git
cd time-series-arima-healthcare

Step 2 — Create a Virtual Environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

Step 3 — Install Dependencies
pip install -r requirements.txt

Step 4 — Launch Jupyter Notebook
jupyter notebook
Open:
notebooks/01_ed_arima_full_walkthrough.ipynb for the complete tutorial
notebooks/00_ed_arima_starter_template.ipynb for the learner exercise version

# 4. Usage Examples
Run the Full Teaching Example
Open in Jupyter:
notebooks/KanishkTawde_TimeSeriesForecastingARIMAHealthcare_WorkingCodeImplementation

This notebook includes:
Data loading and visualization
Stationarity testing (ADF + rolling mean/variance)
Differencing to achieve stationarity
ACF & PACF for parameter selection
ARIMA model training and diagnostics
Forecasting and plotting
Model accuracy evaluation (MAE, RMSE)
Healthcare-focused interpretation

Try the Student Template
Open:
notebooks/KanishkTawde_TimeSeriesForecastingARIMAHealthcare_StarterTemplate

Contains:
Step-by-step TODO cells
Hints for parameter tuning
Debugging guidance
Perfect for learners following the video.

# 5. Learning Objectives

After completing this module, learners will be able to:
📘 Conceptual Understanding
Describe components of a time series (trend, seasonality, residuals)
Explain why stationarity matters for ARIMA
Interpret ACF and PACF plots

🧪 Technical Skills
Preprocess and difference time series data
Fit ARIMA models using statsmodels
Perform residual diagnostics
Compute forecast accuracy metrics

🏥 Applied Healthcare Insight
Translate forecasts into ED staffing implications
Understand uncertainty bounds in operational planning
Identify when ARIMA vs SARIMA vs LSTM is appropriate
These map directly to the evaluation criteria for documentation, pedagogy, and technical mastery.

6. Show-and-Tell Video

A 10-minute video following the Explain → Show → Try teaching model accompanies this repository.

▶ Video Link: https://drive.google.com/file/d/1U_jJqsanzcrIlGdxOnYwNxRn0TLlYBCD/view?usp=sharing

The video includes:
Slide-based conceptual introduction
Live Jupyter Notebook walkthrough
Student practice instructions

7. Requirements:
Requirements
The environment is defined in requirements.txt:

numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.13.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
jupyter>=1.0.0


Install using:
pip install -r requirements.txt

8. Licenses:
Copyright (c) 2025 Kanishk Tawde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


