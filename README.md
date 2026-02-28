# Urban Environmental Intelligence Engine

This repository contains the pipeline and dashboard for the Urban Environmental Intelligence Challenge.

## Setup

1. Create and activate a Python virtual environment.
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. **(Optional)** To use live OpenAQ data instead of synthetic data, set an API key:

- Windows PowerShell:
  ```powershell
  setx OPENAQ_API_KEY "your_real_key_here"
  ```
- macOS/Linux:
  ```bash
  export OPENAQ_API_KEY="your_real_key_here"
  ```

If the variable is not set, the pipeline will run using synthetic data and will print a warning.  
Also note that `api_keys.py` is ignored by default; you'll only need it if you prefer storing the key in a file locally. On Streamlit Cloud, configure the `OPENAQ_API_KEY` in the app settings instead.

## Running

```bash
# run the full pipeline with synthetic data
python main.py

# fetch live data (requires an API key)
python main.py --live

# start the interactive dashboard
streamlit run dashboard.py
```

## Raw API Responses

When live fetching is enabled, JSON responses are saved under `data/raw_openaq/` for auditing and debugging.

---

## Publishing to GitHub & Streamlit ☁️

1. **Prepare `.gitignore`**  
   A template is included in the repository; it ignores the virtual environment, data folders, caches, API keys, and other generated files.

2. **Commit source & docs only**

   ```bash
   git add main.py dashboard.py data_pipeline.py task*.py \
           README.md GETTING_STARTED.md PROJECT_ASSESSMENT.md \
           requirements.txt .gitignore
   git commit -m "Initial project upload"
   git push origin main
   ```

3. **Deploy on Streamlit Community Cloud**
   - Sign in at [streamlit.io/cloud](https://streamlit.io/cloud) and link your GitHub repository.
   - The app command should be: `streamlit run dashboard.py`.
   - Ensure `requirements.txt` is present; Streamlit will install dependencies automatically.
   - Optionally set environment variables (e.g. `OPENAQ_API_KEY`) via the cloud dashboard.

Once the repository is pushed, any changes to the main branch will trigger Streamlit to rebuild and redeploy the app automatically.
