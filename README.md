# Sleep Prediction App

Streamlit GUI for predicting sleep trouble or sleep disorder with the included LightGBM and Logistic Regression pickle models.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Deploy the folder with these files together:

- `app.py`
- `requirements.txt`
- `runtime.txt`
- `lgbm_trouble_model.pkl`
- `logistic_trouble_model.pkl`
- `lgbm_disorder_model.pkl`
- `logistic_disorder_model.pkl`

On Streamlit Community Cloud, select `app.py` as the main file.
