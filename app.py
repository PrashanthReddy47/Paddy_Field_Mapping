import streamlit as st
import ee
from ee import oauth
from google.oauth2 import service_account
import geemap.foliumap as geemap
import pandas as pd
import altair as alt
import folium
from datetime import datetime
import os
import json

# Set page config at the very beginning
st.set_page_config(
    page_title="Paddy Field Mapping and NDVI Analysis",
    page_icon="https://cdn-icons-png.flaticon.com/128/14381/14381464.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/prashanthreddyputta/',
        'Report a bug': "mailto:prashanthputtar@gmail.com",
        'About': "This app demonstrates paddy field mapping using Sentinel-2 imagery and machine learning techniques in the Nalgonda District, India."
    }
)

# Custom CSS
st.markdown("""
<style>
.main {
    scroll-behavior: smooth;
}
.st-emotion-cache-z5fcl4 {
    padding-block: 0;
    position: relative;
}
.st-emotion-cache-16txtl3 {
    padding: 0 1rem;
}
.legend {
    position: absolute;
    top: 10px;
    right: 10px;
    z-index: 1000;
    background-color: rgba(255, 255, 255, 0.8);
    padding: 10px;
    border-radius: 5px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}
.legend h3 {
    margin-top: 0;
    margin-bottom: 5px;
    font-weight: bold;
}
.legend-item {
    display: flex;
    align-items: center;
    margin-bottom: 5px;
}
.legend-color {
    width: 20px;
    height: 20px;
    margin-right: 5px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# Earth Engine authentication function
def ee_authenticate():
    try:
        # Check if running on Streamlit Cloud
        if "gcp_service_account" in st.secrets:
            # Use the service account info from secrets
            service_account_info = st.secrets["gcp_service_account"]
            # Convert the AttrDict to a regular dictionary
            service_account_dict = dict(service_account_info)
            credentials = service_account.Credentials.from_service_account_info(
                service_account_dict,
                scopes=["https://www.googleapis.com/auth/earthengine"]
            )
            ee.Initialize(credentials)
        else:
            # Local development: use ee.Authenticate()
            ee.Authenticate()
            ee.Initialize()
        
        st.success("Earth Engine authenticated successfully.")
    except Exception as e:
        st.error(f"Error authenticating with Earth Engine: {str(e)}")
        st.info("Please check your Earth Engine authentication and try again.")
        st.stop()

# Load Earth Engine assets with error handling
@st.cache_resource
def load_ee_assets():
    try:
        return {
            "Shaligouraram_kattangur_Shapefile": ee.FeatureCollection("projects/ee-unipvgee/assets/Shaligouraram_kattangur_Shapefile"),
            "NDVI_0_65_Threshold": ee.Image("projects/ee-unipvgee/assets/NDVI_Threshold_Rice"),
            "Rice_Field": ee.FeatureCollection("projects/ee-unipvgee/assets/GANESH_AREA"),
            "Classified_RF": ee.Image("projects/ee-unipvgee/assets/RF_Classified_Image"),
            "Classified_SVM": ee.Image("projects/ee-unipvgee/assets/SVM_Classified_Image"),
            "RicePixelsRF": ee.Image("projects/ee-unipvgee/assets/RicePixelsRF"),
            "RicePixelsSVM": ee.Image("projects/ee-unipvgee/assets/RicePixelsSVM")
        }
    except Exception as e:
        st.error(f"Error loading Earth Engine assets: {str(e)}")
        st.info("Please check your asset paths and permissions.")
        st.stop()

# Function to mask clouds in Sentinel-2 imagery
def maskCloudAndShadowsSR(image):
    cloudProb = image.select('MSK_CLDPRB')
    snowProb = image.select('MSK_SNWPRB')
    cloud = cloudProb.lt(10)
    scl = image.select('SCL')
    shadow = scl.eq(3)
    cirrus = scl.eq(10)
    mask = cloud.And(cirrus.Not()).And(shadow.Not())
    return image.updateMask(mask)

# Function to calculate NDVI
def addNDVI(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# Function to get NDVI time series
def get_ndvi_time_series(start_date, end_date):
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(assets["Rice_Field"]) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 26)) \
        .map(maskCloudAndShadowsSR) \
        .map(addNDVI)

    def extract_ndvi(image):
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
        ndvi = image.select('NDVI').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=assets["Rice_Field"],
            scale=20
        ).get('NDVI')
        return ee.Feature(None, {'date': date, 'NDVI': ndvi})

    ndvi_collection = s2.map(extract_ndvi)
    ndvi_values = ndvi_collection.aggregate_array('NDVI').getInfo()
    dates = ndvi_collection.aggregate_array('date').getInfo()
    
    return pd.DataFrame({'date': dates, 'NDVI': ndvi_values})

# Main function to run the Streamlit app
def main():
    # Authenticate Earth Engine
    ee_authenticate()

    # Load Earth Engine assets
    global assets
    assets = load_ee_assets()

    st.title("Paddy Field Mapping using Sentinel-2 Imagery and Machine Learning")
    st.subheader("A Case Study in Nalgonda District, India")

    # Sidebar with app information and user guide
    with st.sidebar:
        st.title("NDVI and Paddy Field Analysis")
        st.markdown(
            """
            ### Navigation:
            - [Map View](#map-view)
            - [NDVI Time Series](#ndvi-time-series-analysis)
            - [Statistics](#ndvi-statistics)
            - [About](#about-this-project)
            
            ### Study Details:
            - **Location**: Nalgonda District, India
            - **Time Period**: Rabi season (Dec 2019 - May 2020)
            - **Data Source**: Sentinel-2 imagery
            - **Methods**: NDVI Thresholding, Random Forest, SVM
            
            ### User Guide:
            1. Select a layer from the dropdown menu to display on the map.
            2. Use the map controls to zoom and pan.
            3. Adjust the layer opacity using the slider.
            4. Set the date range for NDVI time series analysis.
            5. View the NDVI chart and statistics below the map.
            
            ### Contact:
            [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/prashanthreddyputta/)[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:prashanthputtar@gmail.com)
            """
        )

    # Layer options with their respective visualization parameters
    layer_options = {
        "Shaligouraram kattangur Shapefile": {
            "asset": assets["Shaligouraram_kattangur_Shapefile"],
            "vis_params": {"color": "black"},
            "name": "Study Area Boundary"
        },
        "NDVI 0.65 Threshold": {
            "asset": assets["NDVI_0_65_Threshold"].clip(assets["Shaligouraram_kattangur_Shapefile"]),
            "vis_params": {"min": 0, "max": 1, "palette": ['red']},
            "name": "NDVI Threshold Rice Pixels"
        },
        "Random Forest Classification": {
            "asset": assets["Classified_RF"].clip(assets["Shaligouraram_kattangur_Shapefile"]),
            "vis_params": {"min": 0, "max": 4, "palette": ['red', 'cyan', 'green', 'grey', 'blue']},
            "name": "RF Classification"
        },
        "SVM Classification": {
            "asset": assets["Classified_SVM"].clip(assets["Shaligouraram_kattangur_Shapefile"]),
            "vis_params": {"min": 0, "max": 4, "palette": ['red', 'cyan', 'green', 'grey', 'blue']},
            "name": "SVM Classification"
        },
        "Rice Pixels (RF)": {
            "asset": assets["RicePixelsRF"].clip(assets["Shaligouraram_kattangur_Shapefile"]),
            "vis_params": {"min": 0, "max": 1, "palette": ['black']},
            "name": "RF Rice Pixels"
        },
        "Rice Pixels (SVM)": {
            "asset": assets["RicePixelsSVM"].clip(assets["Shaligouraram_kattangur_Shapefile"]),
            "vis_params": {"min": 0, "max": 1, "palette": ['blue']},
            "name": "SVM Rice Pixels"
        }
    }

    selected_layer = st.selectbox("Select a layer to display:", list(layer_options.keys()))

    # Create a map centered on the study area
    Map = geemap.Map(center=[17.252094, 79.323744], zoom=11)

    # Add satellite imagery as a base layer
    Map.add_basemap('HYBRID')

    # Add the selected layer to the map
    layer_info = layer_options[selected_layer]
    layer = Map.addLayer(layer_info["asset"], layer_info["vis_params"], layer_info["name"])

    # Add layer opacity slider
    layer_opacity = st.slider("Layer Opacity", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

    # Update layer opacity
    if layer_opacity < 1.0:
        vis_params = layer_info["vis_params"].copy()
        if "opacity" not in vis_params:
            vis_params["opacity"] = layer_opacity
        Map.addLayer(layer_info["asset"], vis_params, f"{layer_info['name']} (Opacity: {layer_opacity:.1f})")

    # Create Legend
    LEGEND_COLORS = {
        "NDVI 0.65 Threshold": {"Rice": "#FF0000"},
        "Random Forest Classification": {
            "Rice": "#FF0000",
            "Lime/Tangerine": "#00FFFF",
            "Forest/Shrubs": "#008000",
            "Built-Up/Bare Land": "#808080",
            "Water": "#0000FF"
        },
        "SVM Classification": {
            "Rice": "#FF0000",
            "Lime/Tangerine": "#00FFFF",
            "Forest/Shrubs": "#008000",
            "Built-Up/Bare Land": "#808080",
            "Water": "#0000FF"
        },
        "Rice Pixels (RF)": {"Rice Pixels": "#000000"},
        "Rice Pixels (SVM)": {"Rice Pixels": "#0000FF"}
    }

    def create_legend(selected_layer):
        if selected_layer in LEGEND_COLORS:
            legend_html = f"""
            <div id="maplegend" class="maplegend">
                <div class="legend-title">{selected_layer}</div>
                <div class="legend-scale">
                  <ul class="legend-labels">
            """

            for label, color in LEGEND_COLORS[selected_layer].items():
                legend_html += f'<li><span style="background:{color};"></span>{label}</li>'

            legend_html += """
                  </ul>
                </div>
            </div>
            """

            legend_style = """
            <style type='text/css'>
              .maplegend {
                position: absolute;
                z-index:9999;
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 5px;
                border: 2px solid #bbb;
                padding: 10px;
                font-size:12px;
                right: 10px;
                bottom: 20px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
              }
              .maplegend .legend-title {
                text-align: left;
                margin-bottom: 8px;
                font-weight: bold;
                font-size: 14px;
              }
              .maplegend .legend-scale ul {
                margin: 0;
                padding: 0;
                list-style: none;
              }
              .maplegend .legend-scale ul li {
                display: flex;
                align-items: center;
                margin-bottom: 3px;
                font-size: 12px;
              }
              .maplegend ul.legend-labels li span {
                display: block;
                float: left;
                height: 16px;
                width: 24px;
                margin-right: 6px;
                border: 1px solid #999;
              }
            </style>
            """

            return legend_html, legend_style
        return None, None

    # After creating the map and adding layers:
    if selected_layer in LEGEND_COLORS:
        legend_html, legend_style = create_legend(selected_layer)
        if legend_html and legend_style:
            Map.get_root().header.add_child(folium.Element(legend_style))
            Map.get_root().html.add_child(folium.Element(legend_html))

    # Display the map
    Map.to_streamlit(height=600)

    # NDVI Time Series  Analysis
    st.subheader("NDVI Time Series Analysis")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2019, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime(2019, 5, 31))

    if start_date and end_date:
        if start_date < end_date:
            with st.spinner("Calculating NDVI time series..."):
                ndvi_df = get_ndvi_time_series(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            
            # Create NDVI time series chart
            brush = alt.selection_interval(encodings=['x'])
            
            base = alt.Chart(ndvi_df).encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('NDVI:Q', scale=alt.Scale(domain=[0, 1]), title='NDVI')
            )
            
            line = base.mark_line().encode(
                color=alt.condition(brush, alt.value('#4CAF50'), alt.value('lightgray'))
            )
            
            points = base.mark_point().encode(
                color=alt.condition(brush, alt.value('#4CAF50'), alt.value('lightgray')),
                size=alt.condition(brush, alt.value(100), alt.value(30))
            )
            
            chart = (line + points).add_params(brush).properties(
                width=600,
                height=400,
                title='NDVI Time Series of Selected Paddy Field'
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
            # Display statistics
            st.subheader("NDVI Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Average NDVI", f"{ndvi_df['NDVI'].mean():.2f}")
            col2.metric("Median NDVI", f"{ndvi_df['NDVI'].median():.2f}")
            col3.metric("Maximum NDVI", f"{ndvi_df['NDVI'].max():.2f}")
            col4.metric("Minimum NDVI", f"{ndvi_df['NDVI'].min():.2f}")
        else:
            st.error("Error: End date must be after start date.")

    # About This Project
    st.header("About This Project")
    st.write("""
    This app demonstrates paddy field mapping using Sentinel-2 imagery and machine learning techniques in the Nalgonda District, India. 
    It compares NDVI thresholding with Random Forest and SVM classifications to identify and analyze paddy fields in the region.

    **How It Works:**
    1. **Data Collection:** Sentinel-2 imagery is collected for the study area.
    2. **Preprocessing:** Cloud masking and NDVI calculation are performed on the satellite imagery.
    3. **Classification:** Three methods are used to identify paddy fields:  NDVI thresholding, Random Forest, and SVM.
    4. **Analysis:** NDVI time series is generated for the selected paddy fields to monitor crop health and growth.
    5. **Visualization:** Results are displayed on an interactive map and time series chart for easy interpretation.

    **Interpreting the Results:**
    - Higher NDVI values (closer to 1) indicate denser vegetation, typically associated with healthy crops.
    - The time series shows vegetation health over time, useful for monitoring crop growth stages and potential issues.
    - Different classification methods may yield slightly different results due to their underlying algorithms and sensitivity to various factors.
    """)

    # Add footer
    st.markdown("---")
    st.markdown("Created by [Prashanth Reddy Putta](https://www.linkedin.com/in/prashanthreddyputta/) | Data Source: GEE Sentinel-2 Imagery | Powered by Google Earth Engine and Streamlit")

# Run the app
if __name__ == "__main__":
    main()

