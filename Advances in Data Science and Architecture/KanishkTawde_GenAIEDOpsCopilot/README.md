# 🚑 GenAI ED Ops Copilot  
### A Generative AI–Enhanced Forecasting & Decision Support System for Emergency Department Operations  

**Author:** Kanishk Tawde  
**Course:** INFO 7390 — Advances in Data Science & Architecture  
**Instructor:** Prof. Nik Bear Brown  
**Semester:** Fall 2025 

App Link: https://kanishktawde-genai-edops.streamlit.app/

Video Link: https://drive.google.com/file/d/1ua2su09mRNGJgrsHGI1ZgT1VJYFjoWjG/view?usp=sharing

---

# 📘 1. Project Overview

Emergency Departments (EDs) operate under constant uncertainty due to fluctuating patient arrivals, staffing limitations, and unpredictable surges in demand. Although time-series forecasting is routinely applied in healthcare operations, the insights produced by these models are often highly technical and difficult for non-expert hospital leaders to interpret.

The **GenAI ED Ops Copilot** bridges this operational gap by integrating:

- **SARIMAX forecasting** for short-term ED demand prediction  
- **Generative AI (Mistral)** for natural-language interpretability  
- **An interactive Streamlit application** for real-time exploration and Q&A  

This system transforms raw numerical forecasts into **clear, actionable insights** designed to support decision-making in Emergency Departments.

---

# 🎯 2. Project Goals

### **Forecasting**
- Predict daily ED patient volumes using SARIMAX with weekly seasonality.

### **Interpretability**
- Generate natural-language explanations summarizing:
  - Historical trends
  - Forecast patterns  
  - Model uncertainty  
  - Accuracy metrics  

### **Interactive Q&A**
- Allow ED managers to ask open-ended questions such as:
  - “Which days will be highest risk?”
  - “What staffing adjustments should we make next week?”  

### **User Interface**
- Provide a Streamlit web interface so users can:
  - Upload ED visit data  
  - Visualize trends and forecasts  
  - Generate explanations  
  - Ask questions to an AI assistant  

### **Responsible AI**
- Use only synthetic, non-PHI data.  
- Ensure LLM outputs remain grounded in model context.  
- Promote human-in-the-loop decision-making.  

---

# 🏥 3. System Architecture

```
project/
│
├── notebook/
│   └── KanishkTawde_AdvanceDataScience_FinalProject_GenAI-EDOpsCopilot.ipynb
│
├── app/
│   └── app.py
│
├── env/
│   └── .env
│
├── data/
│   └── ed_visits_sample.csv
│
├── requirements.txt
└── README.md
```

---

# 📊 4. Technologies Used

| Component | Technology |
|----------|------------|
| Programming Language | Python 3.10+ |
| Forecasting Model | SARIMAX (Statsmodels) |
| Generative AI | Mistral AI (`mistral-small-latest`) |
| Frontend UI | Streamlit |
| Visualization | Matplotlib, Pandas |
| Environment Management | venv / conda |
| Deployment | Local Streamlit server |

---

# 🧰 5. Setup Instructions

## **5.1 Clone Repository**
```bash
git clone https://github.com/KanishkT04/Kanishk-Tawde-Projects/tree/main/Advances%20in%20Data%20Science%20and%20Architecture/KanishkTawde_GenAIEDOpsCopilot.git
cd KanishkTawde_GenAIEDOpsCopilot
```

## **5.2 Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

## **5.3 Install Dependencies**
```bash
pip install -r requirements.txt
```

## **5.4 Create `.env` File**
Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your_mistral_key_here
```

Get your key at: https://console.mistral.ai  

---

# ▶️ 6. Running the System

## **6.1 Run Streamlit App**
```bash
streamlit run app.py
```
The application opens at:
```
http://localhost:8501
```

---

## **6.2 Run Jupyter Notebook**
```bash
jupyter notebook
```
Open:
```
notebooks/KanishkTawde_AdvanceDataScience_FinalProject_GenAI-EDOpsCopilot.ipynb
```

The notebook includes:
- Data cleaning  
- SARIMAX modeling  
- Forecast generation  
- Evaluation metrics  
- AI explanation generation  
- Q&A system  

---

# 🖥️ 7. Application Workflow

### **7.1 Data Upload**
Users may:
- Upload a CSV file containing daily ED visit counts  
- Or load provided synthetic data  

### **7.2 Historical Visualization**
The app produces:
- Time-series line plot  
- Day-of-week bar chart  
- Basic descriptive statistics  

### **7.3 Forecasting Module**
SARIMAX generates:
- 30-day forecasts  
- Confidence intervals  
- Highlighted risk periods  

### **7.4 Model Evaluation**
The system displays:
- MAE  
- RMSE  
- MAPE  
- Actual vs Predicted curves  

### **7.5 AI-Generated Explanation**
Mistral AI produces:
- Summaries of historical patterns  
- Forecast interpretation  
- Capacity impact insights  
- Staffing recommendations  

### **7.6 Manager Q&A**
Users ask natural-language questions such as:
> “Which days next month require extra staffing?”  
> “How reliable is the forecast for next week?”  

The system answers based strictly on model outputs + context.

---

# 📈 8. Outputs Produced

### ✔ Forecast Plots  
Visual forecast with confidence intervals.

### ✔ Performance Metrics  
MAE, RMSE, and MAPE.

### ✔ LLM Explanations  
Human-friendly summaries including operational perspective.

### ✔ Interactive Q&A  
Generative AI responses grounded in ED forecasting context.

---

# 🧪 9. Testing & Validation

### **Model Validation**
- Tested on a 30-day holdout period  
- Achieved **MAPE ≈ 5%**  
- Accurately captured weekly seasonality  

### **LLM Validation**
- Responses constrained to provided context  
- Outputs verified for clarity, correctness, and lack of hallucination  
- Focus on operational relevance  

---

# ⚖️ 10. Ethical Considerations

- All data used is synthetic and non-identifiable  
- No real patients, hospitals, or PHI  
- LLM outputs restricted to model-derived context  
- Tool is positioned as *decision support*, not autonomous decision-making  
- Encourages human oversight  

---

# 🧭 11. Future Enhancements

Potential improvements:

- Integrate external features like weather, flu season, holidays  
- Extend forecast horizon using hybrid LSTM models  
- Add anomaly detection for ED surges  
- Deploy to Streamlit Cloud for public use  
- Implement staffing optimization recommendations  
- Add RAG-based retrieval for operational policies  

---

# 🎥 12. Video Presentation

Video Link: https://drive.google.com/file/d/1ua2su09mRNGJgrsHGI1ZgT1VJYFjoWjG/view?usp=sharing

---

# 👤 13. Contributor

**Kanishk Tawde**  
Master of Science in Information Systems  
Northeastern University  

---

# ✔️ 14. License

Copyright (c) 2025 Kanishk Tawde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---


This project demonstrates how **forecasting + generative AI** can create powerful decision-support tools that improve operational readiness in Emergency Departments.

