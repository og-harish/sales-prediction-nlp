# ⚡ Sales Prediction System with NLP-Based Insight Extraction and Dashboard

An intelligent, end-to-end business intelligence (BI) and predictive forecasting suite. This application accepts raw uncleaned retail sales records, automatically cleans and formats them, executes multiple machine learning models to identify the best forecaster, runs future transactional predictions, extracts strategic business insights via Generative AI (using Llama 3 via Groq API), and compiles downloadable executive PDF reports.

Designed with a **sleek, glassmorphic dark-theme UI** featuring glowing indicators, responsive analytics grids, and robust offline fallbacks, this project represents a production-ready, portfolio-grade system.

---

## 🚀 Key Features

1. **Automated Cleaning Pipeline**: Automatically parses dates, imputes missing values (median for numeric, mode for categorical), handles duplicates, and normalizes messy text categories (e.g. standardizing raw product inputs).
2. **Deterministic Feature Engineering**: Creates price categories (Low/Medium/High), extracts temporal parameters (quarters, weekends, seasons), and computes customer sentiment polarity.
3. **Automatic ML Model Selector**: Trains and compares three standard regressors (**Linear Regression**, **Decision Tree**, and **Random Forest Regressor**) using $R^2$ Score, MAE, and RMSE. Automatically registers and serializes the highest performing model.
4. **Vibrant Glassmorphic Frontend**: Streamlit user interface featuring interactive Plotly visualization grids, responsive card containers, custom gradients, and smooth animations.
5. **Generative AI Strategy Engine**: Queries Llama 3 on Groq to extract business summaries, positive findings, risks, and recommendations. Supports an advanced analytical **rule-based local fallback engine** if the network is offline or no API key is set.
6. **Executive PDF Exporter**: Exporters for both cleaned CSV files and fully formatted PDF summaries (which automatically map special characters like the Indian Rupee symbol `₹` to `INR` to ensure crash-free document compiling).

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend & ML**: Python, Scikit-Learn, Pandas, NumPy, Joblib
- **Exploratory Data Analysis (EDA)**: Plotly, Matplotlib
- **Natural Language Processing (NLP)**: TextBlob, Groq SDK (Llama 3.3 model)
- **Document Exporting**: FPDF
- **Environment Management**: `uv` package manager (optional, but highly recommended)

---

## 📂 Folder Structure

```
sales-prediction-nlp/
│
├── data/
│   ├── Flipkart_Sales_Uncleaned.csv        # Primary raw data source
│   └── Flipkart_Sales_Cleaned.csv          # Generated after preprocessing
│
├── models/
│   └── sales_model.pkl                      # Serialized best model state & encoder mappings
│
├── notebooks/
│   └── preprocessing.ipynb                  # step-by-step explanatory Jupyter tutorial
│
├── pages/
│   ├── dashboard.py                        # Metric KPI cards & 7 interactive Plotly charts
│   ├── prediction.py                       # Future prediction inputs & confidence score panels
│   ├── insights.py                         # Sentiment gauge charts & Groq NLP strategic insights
│   └── reports.py                          # In-memory CSV and PDF executive report exporters
│
├── utils/
│   ├── preprocess.py                       # Imputation, parsing, and feature engineering logic
│   ├── model.py                            # ML model training, comparisons, and inference handlers
│   ├── nlp.py                              # TextBlob sentiment & Groq API LLM wrapper
│   └── charts.py                           # Customized transparent Plotly chart blueprints
│
├── app.py                                  # Primary welcome hub and pipeline trigger portal
├── requirements.txt                        # Core package dependencies
└── README.md                               # Project documentation
```

---

## 💻 Installation & Local Setup

### Option A: Setup using Astral `uv` (Recommended - Ultra Fast)

`uv` downloads portable python on-demand and sets up environments instantly without global conflicts.

1. **Install `uv`** (if not already installed):
   * **Windows (PowerShell)**:
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   * **Linux/macOS**:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

2. **Run Streamlit immediately**:
   ```bash
   uv run streamlit run app.py
   ```
   *(This will automatically create a virtual environment, install all requirements, and launch the application!)*

---

### Option B: Standard Python `pip` Setup

1. **Clone or navigate to the workspace directory**:
   ```bash
   cd sales-prediction-nlp
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   * **Windows (Cmd)**: `.venv\Scripts\activate.bat`
   * **Windows (PowerShell)**: `.venv\Scripts\Activate.ps1`
   * **Linux/macOS**: `source .venv/bin/activate`

4. **Install core dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

---

## 📋 Comprehensive Usage Workflow

1. **Home / Welcoming Hub (`app.py`)**:
   * Inspect raw dataset details and columns.
   * Upload a custom raw sales dataset or use the pre-loaded uncleaned Flipkart sales file.
   * Click **🧼 Clean & Process Dataset**. A progress bar will show real-time logs while cleaning data and fitting Linear Regression, Decision Tree, and Random Forest models.
   * View the training comparison matrix and verify that the best-performing model has been automatically selected and saved.

2. **Analytics Dashboard (`pages/dashboard.py`)**:
   * View KPI metrics for Total Revenue, Volume, Top Product Category, Top Sales State, and Average Discount.
   * Interact with **7 custom visualizations**: Sales Performance Line, State Contribution Bar, Revenue Share Donut, Aggregated Monthly Line, Feature Correlation Heatmap, Top Products Horizontal Bar, and Discount Scatter plots.

3. **Forecast Forecasting (`pages/prediction.py`)**:
   * Select a hypothetical Product, State, and City (dropdowns populated dynamically from active values).
   * Specify Quantity volume, Unit pricing, discount rates, and target dates.
   * Predict the future transaction value, displayed in a glowing, formatted currency display alongside the model's R²-based confidence score.

4. **NLP Insights (`pages/insights.py`)**:
   * Examine transaction sentiment polarity metrics and explore a custom sentiment gauge chart.
   * Click **🧠 Extract AI Business Insights** to fetch live strategic reports from Groq (Llama 3) covering business summaries, trend analysis, positive discoveries, risks, and recommendations.

5. **Document Exporters (`pages/reports.py`)**:
   * Download the complete, cleaned dataset as a CSV.
   * Export and compile a professional PDF report containing the company letterhead, transactional KPI tables, and AI NLP findings.

---

## 🌐 Production Cloud Deployment

### 1. Deploying on Streamlit Community Cloud (Recommended)
1. Push this project folder directly to a **GitHub Repository**.
2. Visit [Streamlit Share](https://share.streamlit.io/) and log in with your GitHub credentials.
3. Click **New App**, select your repository, branch (`main`), and set the main file path to `app.py`.
4. Expand **Advanced Settings** and inject your Groq API Key:
   ```toml
   GROQ_API_KEY = "your-gsk-key-here"
   ```
5. Click **Deploy!** Your app will be live on a public URL in minutes.

### 2. Deploying on Render (Docker or Web Service)
1. Add a file named `Dockerfile` in the root of the project:
   ```dockerfile
   FROM python:3.10-slim
   EXPOSE 8501
   WORKDIR /app
   COPY . .
   RUN pip install --no-cache-dir -r requirements.txt
   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```
2. Log in to [Render](https://render.com/) and create a new **Web Service**.
3. Link your GitHub repository.
4. Set the runtime environment to **Docker**.
5. Under **Environment Variables**, add:
   * Key: `GROQ_API_KEY`, Value: `your-gsk-key-here`
6. Click **Deploy Web Service**. Render will build and host your service.
