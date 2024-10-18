# Paddy Field Mapping using Sentinel-2 Imagery NDVI and Machine Learning

This Streamlit app demonstrates paddy field mapping using Sentinel-2 imagery and machine learning techniques in the Nalgonda District, India. It compares NDVI thresholding with Random Forest and SVM classifications to identify and analyze paddy fields in the region.

## Features

- Interactive map displaying various layers (NDVI threshold, Random Forest, SVM classifications)
- NDVI time series analysis
- Statistical information about NDVI values

## Installation

1. Clone this repository:
   ```
   git clone
   cd your-repo-name
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up Earth Engine authentication:
   - For local development, use `ee.Authenticate()` in the Python console.
   - For deployment, set up a service account and add the credentials to Streamlit secrets.

## Usage

Run the Streamlit app locally:

```
streamlit run app.py
```

## Deployment

This app is designed to be deployed on Streamlit Community Cloud. Follow these steps:

1. Push your code to GitHub (excluding `secrets.toml`).
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and click "New app".
3. Connect your GitHub repo and select the main file (app.py).
4. In "Advanced settings", paste your Earth Engine service account credentials.
5. Deploy the app.

## Feedback

Any feedback is welcome and appreciated.

## License

This project is licensed under the MIT License.