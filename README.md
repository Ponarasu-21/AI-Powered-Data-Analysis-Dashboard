# AI-Powered Data Analysis Dashboard

This project is a Streamlit web application for data preprocessing, visualization, correlation analysis, and basic AI insights.

## Task Objective
Build a web-based dashboard that:
- Imports a CSV dataset
- Displays dataset statistics
- Handles missing values
- Generates visualizations
- Shows correlations between features
- Provides basic AI-generated insights

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- scikit-learn

## Files
- `app.py` - Streamlit dashboard implementation
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `.gitignore` - ignores environment and cache files

## Getting Started
1. Open the workspace in VS Code.
2. Create and activate a Python virtual environment.
   - Windows PowerShell:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```powershell
   streamlit run app.py
   ```

## Using the Dashboard
1. Upload a CSV dataset from the sidebar.
2. Review dataset statistics and preview the first rows.
3. Explore missing-value handling options.
4. View correlation heatmaps and visualizations.
5. Select a target column for basic AI/ML model evaluation.

## Deployment
Deploy to Streamlit Community Cloud:
1. Push the repository to GitHub. ✅ **Done:** https://github.com/Ponarasu-21/AI-Powered-Data-Analysis-Dashboard
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io).
3. Connect the GitHub repository and deploy.
4. Copy the generated deployment URL and paste below.

## Sample Dataset
To test the dashboard, you can use any CSV file with:
- Numeric columns (for correlation analysis and regression)
- Categorical columns (for classification tasks)
- Some missing values (to test preprocessing)

Example: Use datasets from [Kaggle](https://kaggle.com), [UCI ML Repository](https://archive.ics.uci.edu/ml/), or create a sample CSV.

## Deliverables
- **Live Streamlit link:** https://ai-powered-data-analysis-dashboard-iksr2vpjnxvtcuuc2vvadv.streamlit.app/
- **GitHub repository:** https://github.com/Ponarasu-21/AI-Powered-Data-Analysis-Dashboard
- **Dataset used:** Employee Salary Dataset (sample_data.csv)
  - 20 employee records with Age, Salary, Experience, Department, and Bonus_Eligible columns
- **LinkedIn post:** [Add your LinkedIn post URL or summary here]
  - Share your learning journey and key insights from building this dashboard

## Key Features Implemented
✅ CSV dataset upload and preview  
✅ Missing value handling (multiple strategies)  
✅ Correlation heatmap visualization  
✅ Distribution and scatter plot charts  
✅ AI-generated dataset insights  
✅ Basic ML model evaluation (regression/classification)  
✅ Responsive Streamlit UI
