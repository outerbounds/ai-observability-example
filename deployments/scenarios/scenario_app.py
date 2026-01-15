"""
Wildfire Destruction Prediction Scenario Builder

A Streamlit app that uses the destruction-prediction model from WildfireFlow
to estimate the likelihood of structure destruction based on building characteristics
and location.
"""

import streamlit as st
import numpy as np
from metaflow import Flow, namespace
from streamlit_folium import st_folium

# California counties with approximate centroid coordinates
CALIFORNIA_COUNTIES = {
    "Alameda": (37.6017, -121.7195),
    "Amador": (38.4463, -120.6542),
    "Butte": (39.6670, -121.6008),
    "Calaveras": (38.1877, -120.5592),
    "Colusa": (39.1776, -122.2375),
    "Contra Costa": (37.9193, -121.9277),
    "El Dorado": (38.7786, -120.5246),
    "Fresno": (36.7585, -119.6482),
    "Glenn": (39.5983, -122.3922),
    "Inyo": (36.5115, -117.4109),
    "Kern": (35.3429, -118.7295),
    "Lake": (39.0997, -122.7536),
    "Lassen": (40.6736, -120.5964),
    "Los Angeles": (34.3083, -118.2280),
    "Madera": (37.2181, -119.7631),
    "Mariposa": (37.5831, -119.9665),
    "Mendocino": (39.4337, -123.3913),
    "Monterey": (36.2160, -121.2495),
    "Napa": (38.5025, -122.2654),
    "Nevada": (39.3013, -120.7689),
    "Orange": (33.7175, -117.8311),
    "Placer": (39.0634, -120.7176),
    "Plumas": (40.0034, -120.8389),
    "Riverside": (33.7437, -115.9939),
    "Sacramento": (38.4500, -121.3400),
    "San Benito": (36.6058, -121.0750),
    "San Bernardino": (34.8414, -116.1781),
    "San Diego": (33.0284, -116.7679),
    "San Joaquin": (37.9349, -121.2716),
    "San Luis Obispo": (35.3872, -120.4522),
    "San Mateo": (37.4337, -122.4014),
    "Santa Barbara": (34.5374, -120.0388),
    "Santa Clara": (37.2319, -121.6951),
    "Santa Cruz": (37.0603, -122.0067),
    "Shasta": (40.7637, -122.0403),
    "Siskiyou": (41.5926, -122.5402),
    "Solano": (38.2668, -121.9400),
    "Sonoma": (38.5254, -122.9276),
    "Stanislaus": (37.5593, -120.9971),
    "Tehama": (40.1257, -122.2343),
    "Trinity": (40.6510, -123.1117),
    "Tulare": (36.2278, -118.7815),
    "Tuolumne": (38.0272, -119.9545),
    "Ventura": (34.4583, -119.0322),
    "Yolo": (38.6864, -121.9018),
    "Yuba": (39.2678, -121.3500),
}

# Feature options based on training data
FEATURE_OPTIONS = {
    "structure_type": [
        "Single Family Residence Single Story",
        "Single Family Residence Multi Story",
        "Mobile Home Single Wide",
        "Mobile Home Double Wide",
        "Mobile Home Triple Wide",
        "Multi Family Residence Single Story",
        "Multi Family Residence Multi Story",
        "Mixed Commercial/Residential",
        "Commercial Building Single Story",
        "Commercial Building Multi Story",
        "Motor Home",
        "Utility Misc Structure",
        "Agriculture",
        "Infrastructure",
        "Church",
        "School",
        "Hospital",
    ],
    "structure_category": [
        "Single Residence",
        "Multiple Residence",
        "Mixed Commercial/Residential",
        "Nonresidential Commercial",
        "Other Minor Structure",
        "Infrastructure",
        "Agriculture",
    ],
    "roof_construction": [
        "Asphalt",
        "Metal",
        "Tile",
        "Concrete",
        "Wood",
        "Other",
        "Unknown",
    ],
    "eaves": [
        "Enclosed",
        "Unenclosed",
        "No Eaves",
        "Unknown",
    ],
    "vent_screen": [
        'Mesh Screen <= 1/8"',
        'Mesh Screen > 1/8"',
        "Unscreened",
        "No Vents",
        "Unknown",
    ],
    "exterior_siding": [
        "Stucco/Brick/Cement",
        "Ignition Resistant",
        "Wood",
        "Metal",
        "Vinyl",
        "Combustible",
        "Other",
        "Unknown",
    ],
    "window_pane": [
        "Multi Pane",
        "Single Pane",
        "Radiant Heat",
        "No Windows",
        "Unknown",
    ],
    "deck_on_grade": [
        "No Deck/Porch",
        "Wood",
        "Composite",
        "Masonry/Concrete",
        "Unknown",
    ],
    "deck_elevated": [
        "No Deck/Porch",
        "Wood",
        "Composite",
        "Masonry/Concrete",
        "Unknown",
    ],
    "patio_cover": [
        "No Patio Cover/Carport",
        "Non Combustible",
        "Combustible",
        "Unknown",
    ],
    "fence_attached": [
        "No Fence",
        "Non Combustible",
        "Combustible",
        "Unknown",
    ],
}

FEATURE_LABELS = {
    "structure_type": "Structure Type",
    "structure_category": "Structure Category",
    "roof_construction": "Roof Construction",
    "eaves": "Eaves Type",
    "vent_screen": "Vent Screen",
    "exterior_siding": "Exterior Siding",
    "window_pane": "Window Pane Type",
    "deck_on_grade": "Deck/Porch On Grade",
    "deck_elevated": "Deck/Porch Elevated",
    "patio_cover": "Patio Cover/Carport",
    "fence_attached": "Fence Attached",
}


@st.cache_resource
def load_model():
    """Load the trained model from the latest successful WildfireFlow run."""
    try:
        namespace(None)
        flow = Flow("WildfireFlow")
        run = flow.latest_successful_run

        if run is None:
            st.error("No successful WildfireFlow run found. Please run the flow first.")
            return None, None, None, None, None

        train_step = run["train"].task
        model = train_step.data.model
        encoders = train_step.data.encoders
        feature_cols = train_step.data.feature_cols
        auc_score = train_step.data.auc_score
        run_id = run.id

        return model, encoders, feature_cols, auc_score, run_id
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None


def encode_feature(value, encoder, feature_name):
    """Encode a feature value using the fitted encoder."""
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # Value not seen during training, use "Unknown" if available
        try:
            return encoder.transform(["Unknown"])[0]
        except ValueError:
            # Fall back to first class
            return 0


def predict_destruction(model, encoders, feature_cols, features):
    """Make a prediction using the model."""
    encoded_features = []
    for col in feature_cols:
        value = features.get(col, "Unknown")
        encoded = encode_feature(value, encoders[col], col)
        encoded_features.append(encoded)

    X = np.array([encoded_features])
    proba = model.predict_proba(X)[0, 1]
    return proba


def create_county_map(selected_county):
    """Create an interactive California counties map."""
    import folium

    # California center
    ca_center = [37.0, -119.5]

    m = folium.Map(location=ca_center, zoom_start=6, tiles="cartodbpositron")

    # Add county markers
    for county, (lat, lon) in CALIFORNIA_COUNTIES.items():
        is_selected = county == selected_county
        folium.CircleMarker(
            location=[lat, lon],
            radius=12 if is_selected else 8,
            popup=county,
            tooltip=county,
            color="#d62728" if is_selected else "#1f77b4",
            fill=True,
            fillColor="#d62728" if is_selected else "#1f77b4",
            fillOpacity=0.7 if is_selected else 0.4,
        ).add_to(m)

    return m


def main():
    st.set_page_config(
        page_title="Wildfire Destruction Predictor",
        page_icon="🔥",
        layout="wide",
    )

    st.title("Wildfire Destruction Scenario Builder")
    st.markdown("""
    This tool predicts the likelihood of structure destruction during a wildfire
    based on building characteristics and location. The model was trained on
    California wildfire incident data.
    """)

    # Load model
    model, encoders, feature_cols, auc_score, run_id = load_model()

    if model is None:
        st.stop()

    # Display Run ID prominently
    st.info(f"**Model Run ID:** `{run_id}`")

    st.sidebar.markdown(f"**Model Run ID:** `{run_id}`")
    st.sidebar.markdown(f"**Model AUC Score:** {auc_score:.3f}")

    # Layout: map on left, features on right
    col_map, col_features = st.columns([1, 1])

    with col_map:
        st.subheader("Select County")

        # County dropdown (also selectable from map)
        selected_county = st.selectbox(
            "Choose a county:",
            options=sorted(CALIFORNIA_COUNTIES.keys()),
            index=sorted(CALIFORNIA_COUNTIES.keys()).index("Los Angeles"),
        )

        # Interactive map
        try:
            m = create_county_map(selected_county)
            map_data = st_folium(
                m,
                width=500,
                height=400,
                returned_objects=["last_object_clicked"],
            )

            # Update county if user clicked on map
            if map_data and map_data.get("last_object_clicked"):
                clicked = map_data["last_object_clicked"]
                if clicked:
                    # Find closest county to click
                    click_lat = clicked.get("lat")
                    click_lng = clicked.get("lng")
                    if click_lat and click_lng:
                        min_dist = float("inf")
                        closest_county = selected_county
                        for county, (lat, lon) in CALIFORNIA_COUNTIES.items():
                            dist = (lat - click_lat) ** 2 + (lon - click_lng) ** 2
                            if dist < min_dist:
                                min_dist = dist
                                closest_county = county
                        if closest_county != selected_county and min_dist < 0.5:
                            st.rerun()
        except ImportError:
            st.warning("Install `streamlit-folium` and `folium` for interactive map.")
            st.info(f"Selected county: **{selected_county}**")

    with col_features:
        st.subheader("Building Characteristics")

        features = {"county": selected_county}

        # Create two columns for feature selection
        feat_col1, feat_col2 = st.columns(2)

        feature_items = list(FEATURE_OPTIONS.items())
        half = len(feature_items) // 2

        with feat_col1:
            for feat_name, options in feature_items[:half]:
                features[feat_name] = st.selectbox(
                    FEATURE_LABELS[feat_name],
                    options=options,
                    key=feat_name,
                )

        with feat_col2:
            for feat_name, options in feature_items[half:]:
                features[feat_name] = st.selectbox(
                    FEATURE_LABELS[feat_name],
                    options=options,
                    key=feat_name,
                )

    # Prediction section
    st.markdown("---")

    if st.button("Predict Destruction Likelihood", type="primary", use_container_width=True):
        with st.spinner("Calculating..."):
            probability = predict_destruction(model, encoders, feature_cols, features)
        st.session_state["prediction_result"] = probability
        st.session_state["prediction_features"] = features.copy()

    # Display result if available
    if "prediction_result" in st.session_state:
        probability = st.session_state["prediction_result"]

        # Color based on risk level
        if probability < 0.3:
            risk_level = "Low Risk"
            risk_emoji = "🟢"
        elif probability < 0.6:
            risk_level = "Moderate Risk"
            risk_emoji = "🟡"
        else:
            risk_level = "High Risk"
            risk_emoji = "🔴"

        st.subheader("Prediction Result")

        result_col1, result_col2, result_col3 = st.columns([1, 2, 1])

        with result_col2:
            st.metric(
                label="Destruction Likelihood",
                value=f"{probability:.1%}",
                delta=f"{risk_emoji} {risk_level}",
                delta_color="off"
            )
            st.caption("Likelihood of structure destruction (>50% damage)")

            # Show feature summary
            with st.expander("View scenario details"):
                st.write("**Selected Features:**")
                saved_features = st.session_state.get("prediction_features", {})
                for col in feature_cols:
                    label = FEATURE_LABELS.get(col, col.replace("_", " ").title())
                    st.write(f"- {label}: {saved_features.get(col, 'Unknown')}")

    # Footer
    st.markdown("---")
    st.caption(
        "Model trained on California wildfire structure damage data. "
        "Predictions are estimates based on historical patterns and should not be "
        "used as the sole basis for safety decisions."
    )


if __name__ == "__main__":
    main()
