from __future__ import annotations

import json
import pickle
from pathlib import Path

import h3
import joblib
import numpy as np
import pandas as pd
import altair as alt
import pydeck as pdk
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "traffic_risk_model.pkl"
HEX_RISK_PATH = PROJECT_ROOT / "hex_risk_dict.pkl"
FEATURE_ORDER_PATH = PROJECT_ROOT / "feature_order.json"
WEATHER_CATEGORIES_PATH = PROJECT_ROOT / "weather_categories.json"
ACCENT = "#DD5A3C"
INK = "#14212B"
GOLD = "#D9A441"
GREEN = "#2E8B57"
DEEP = "#8C2F1C"

STUDY_AREAS = {
    "Los Angeles Core": {
        "lat": 34.0522,
        "lng": -118.2437,
        "focus": "dense urban commute patterns, junction-heavy road geometry, and high historical neighborhood incident density",
    },
    "Bay Area Commute": {
        "lat": 37.7749,
        "lng": -122.4194,
        "focus": "fog exposure, morning commute timing, and corridor congestion around major city connectors",
    },
    "Central Valley Corridor": {
        "lat": 36.7783,
        "lng": -119.4179,
        "focus": "lower-density baseline travel conditions for comparison against coastal metro risk patterns",
    },
}


st.set_page_config(page_title="California Traffic Risk Engine", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with HEX_RISK_PATH.open("rb") as handle:
        risk_dict = pickle.load(handle)
    saved_feature_order = json.loads(FEATURE_ORDER_PATH.read_text())
    model_feature_order = list(getattr(model, "feature_name_", []) or getattr(model, "feature_names_in_", []))
    feature_order = model_feature_order if model_feature_order else saved_feature_order
    weather_categories = json.loads(WEATHER_CATEGORIES_PATH.read_text())
    return model, risk_dict, feature_order, weather_categories


def build_feature_frame(
    lat: float,
    lng: float,
    hour: int,
    temp: float,
    hum: float,
    vis: float,
    wind: float,
    weather_sel: str,
    light_sel: str,
    crossing: int,
    junction: int,
    stop: int,
    traffic_signal: int,
    station: int,
    risk_dict: dict[str, float],
    feature_order: list[str],
    weather_categories: list[str],
) -> tuple[pd.DataFrame, str, float]:
    hex_id = h3.latlng_to_cell(lat, lng, 7)
    raw_neighbor_risk = float(sum(risk_dict.get(cell, 0) for cell in h3.grid_disk(hex_id, 1)))
    neighbor_risk = float(np.log1p(raw_neighbor_risk))

    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))
    wet_rush_hour = float((hum / 100.0) * hour_sin)

    frame = pd.DataFrame(
        [
            {
                "Temperature(F)": float(temp),
                "Humidity(%)": float(hum),
                "Visibility(mi)": float(vis),
                "Wind_Speed(mph)": float(wind),
                "Weather_Condition": weather_sel,
                "Sunrise_Sunset": light_sel,
                "Crossing": float(crossing),
                "Junction": float(junction),
                "Stop": float(stop),
                "Traffic_Signal": float(traffic_signal),
                "Station": float(station),
                "neighbor_risk": neighbor_risk,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "wet_rush_hour": wet_rush_hour,
            }
        ]
    )

    frame["Weather_Condition"] = pd.Categorical(frame["Weather_Condition"], categories=weather_categories)
    frame["Sunrise_Sunset"] = pd.Categorical(frame["Sunrise_Sunset"], categories=["Day", "Night"])
    frame = frame.reindex(columns=feature_order)
    return frame, hex_id, raw_neighbor_risk


def validate_assets() -> None:
    missing = [
        path.name
        for path in [MODEL_PATH, HEX_RISK_PATH, FEATURE_ORDER_PATH, WEATHER_CATEGORIES_PATH]
        if not path.exists()
    ]
    if missing:
        st.error(f"Missing required dashboard files: {', '.join(missing)}")
        st.stop()


def predict_probability(model, input_df: pd.DataFrame) -> float:
    return float(model.predict_proba(input_df)[0][1])


@st.cache_data(show_spinner=False)
def build_hex_map_df(risk_dict: dict[str, float], top_n: int = 1200) -> pd.DataFrame:
    rows = []
    for hex_id, incidents in sorted(risk_dict.items(), key=lambda item: item[1], reverse=True)[:top_n]:
        lat, lng = h3.cell_to_latlng(hex_id)
        alpha = int(min(220, max(40, incidents / 25)))
        rows.append(
            {
                "hex_id": hex_id,
                "lat": float(lat),
                "lng": float(lng),
                "incidents": int(incidents),
                "log_risk": float(np.log1p(incidents)),
                "color_r": 221,
                "color_g": 90,
                "color_b": 60,
                "color_a": alpha,
            }
        )
    return pd.DataFrame(rows)


def make_scenario(
    *,
    lat: float,
    lng: float,
    hour: int,
    temp: float,
    hum: float,
    vis: float,
    wind: float,
    weather_sel: str,
    light_sel: str,
    crossing: int,
    junction: int,
    stop: int,
    traffic_signal: int,
    station: int,
    risk_dict: dict[str, float],
    feature_order: list[str],
    weather_categories: list[str],
) -> tuple[pd.DataFrame, str, float]:
    return build_feature_frame(
        lat=lat,
        lng=lng,
        hour=hour,
        temp=temp,
        hum=hum,
        vis=vis,
        wind=wind,
        weather_sel=weather_sel,
        light_sel=light_sel,
        crossing=crossing,
        junction=junction,
        stop=stop,
        traffic_signal=traffic_signal,
        station=station,
        risk_dict=risk_dict,
        feature_order=feature_order,
        weather_categories=weather_categories,
    )


def build_benchmark_df(
    model,
    risk_dict: dict[str, float],
    feature_order: list[str],
    weather_categories: list[str],
    current_inputs: dict[str, float | int | str],
) -> pd.DataFrame:
    scenarios = [
        {
            "name": "Current Scenario",
            **current_inputs,
        },
        {
            "name": "Clear Midday Baseline",
            "lat": 36.7783,
            "lng": -119.4179,
            "hour": 11,
            "temp": 72,
            "hum": 35,
            "vis": 10.0,
            "wind": 4,
            "weather_sel": "Clear" if "Clear" in weather_categories else weather_categories[0],
            "light_sel": "Day",
            "crossing": 0,
            "junction": 0,
            "stop": 0,
            "traffic_signal": 0,
            "station": 0,
        },
        {
            "name": "LA Rain Commute",
            "lat": 34.0522,
            "lng": -118.2437,
            "hour": 8,
            "temp": 58,
            "hum": 95,
            "vis": 0.5,
            "wind": 28,
            "weather_sel": "Rain" if "Rain" in weather_categories else weather_categories[0],
            "light_sel": "Day",
            "crossing": 1,
            "junction": 1,
            "stop": 1,
            "traffic_signal": 1,
            "station": 1,
        },
        {
            "name": "Bay Fog Commute",
            "lat": 37.7749,
            "lng": -122.4194,
            "hour": 8,
            "temp": 54,
            "hum": 92,
            "vis": 0.8,
            "wind": 18,
            "weather_sel": "Fog" if "Fog" in weather_categories else weather_categories[0],
            "light_sel": "Day",
            "crossing": 1,
            "junction": 1,
            "stop": 0,
            "traffic_signal": 1,
            "station": 0,
        },
    ]

    rows = []
    for scenario in scenarios:
        frame, hex_id, raw_neighbor_risk = make_scenario(
            lat=float(scenario["lat"]),
            lng=float(scenario["lng"]),
            hour=int(scenario["hour"]),
            temp=float(scenario["temp"]),
            hum=float(scenario["hum"]),
            vis=float(scenario["vis"]),
            wind=float(scenario["wind"]),
            weather_sel=str(scenario["weather_sel"]),
            light_sel=str(scenario["light_sel"]),
            crossing=int(scenario["crossing"]),
            junction=int(scenario["junction"]),
            stop=int(scenario["stop"]),
            traffic_signal=int(scenario["traffic_signal"]),
            station=int(scenario["station"]),
            risk_dict=risk_dict,
            feature_order=feature_order,
            weather_categories=weather_categories,
        )
        rows.append(
            {
                "scenario": scenario["name"],
                "probability": predict_probability(model, frame),
                "hour": int(scenario["hour"]),
                "weather": str(scenario["weather_sel"]),
                "neighbor_incidents": int(raw_neighbor_risk),
                "hex_id": hex_id,
            }
        )
    return pd.DataFrame(rows)


def build_hourly_profile_df(
    model,
    risk_dict: dict[str, float],
    feature_order: list[str],
    weather_categories: list[str],
    current_inputs: dict[str, float | int | str],
) -> pd.DataFrame:
    rows = []
    for hour in range(24):
        frame, _, _ = make_scenario(
            lat=float(current_inputs["lat"]),
            lng=float(current_inputs["lng"]),
            hour=hour,
            temp=float(current_inputs["temp"]),
            hum=float(current_inputs["hum"]),
            vis=float(current_inputs["vis"]),
            wind=float(current_inputs["wind"]),
            weather_sel=str(current_inputs["weather_sel"]),
            light_sel=str(current_inputs["light_sel"]),
            crossing=int(current_inputs["crossing"]),
            junction=int(current_inputs["junction"]),
            stop=int(current_inputs["stop"]),
            traffic_signal=int(current_inputs["traffic_signal"]),
            station=int(current_inputs["station"]),
            risk_dict=risk_dict,
            feature_order=feature_order,
            weather_categories=weather_categories,
        )
        rows.append({"hour": hour, "probability": predict_probability(model, frame)})
    return pd.DataFrame(rows)


def build_weather_hour_heatmap_df(
    model,
    risk_dict: dict[str, float],
    feature_order: list[str],
    weather_categories: list[str],
    current_inputs: dict[str, float | int | str],
) -> pd.DataFrame:
    preferred_weather = ["Clear", "Cloudy", "Fog", "Rain"]
    selected_weather = [w for w in preferred_weather if w in weather_categories]
    if not selected_weather:
        selected_weather = weather_categories[:4]

    rows = []
    for weather in selected_weather:
        for hour in range(24):
            frame, _, _ = make_scenario(
                lat=float(current_inputs["lat"]),
                lng=float(current_inputs["lng"]),
                hour=hour,
                temp=float(current_inputs["temp"]),
                hum=float(current_inputs["hum"]),
                vis=float(current_inputs["vis"]),
                wind=float(current_inputs["wind"]),
                weather_sel=str(weather),
                light_sel=str(current_inputs["light_sel"]),
                crossing=int(current_inputs["crossing"]),
                junction=int(current_inputs["junction"]),
                stop=int(current_inputs["stop"]),
                traffic_signal=int(current_inputs["traffic_signal"]),
                station=int(current_inputs["station"]),
                risk_dict=risk_dict,
                feature_order=feature_order,
                weather_categories=weather_categories,
            )
            rows.append(
                {
                    "hour": hour,
                    "weather": weather,
                    "probability": predict_probability(model, frame),
                }
            )
    return pd.DataFrame(rows)


def build_visibility_sensitivity_df(
    model,
    risk_dict: dict[str, float],
    feature_order: list[str],
    weather_categories: list[str],
    current_inputs: dict[str, float | int | str],
) -> pd.DataFrame:
    rows = []
    for visibility in np.linspace(0.5, 10.0, 12):
        frame, _, _ = make_scenario(
            lat=float(current_inputs["lat"]),
            lng=float(current_inputs["lng"]),
            hour=int(current_inputs["hour"]),
            temp=float(current_inputs["temp"]),
            hum=float(current_inputs["hum"]),
            vis=float(visibility),
            wind=float(current_inputs["wind"]),
            weather_sel=str(current_inputs["weather_sel"]),
            light_sel=str(current_inputs["light_sel"]),
            crossing=int(current_inputs["crossing"]),
            junction=int(current_inputs["junction"]),
            stop=int(current_inputs["stop"]),
            traffic_signal=int(current_inputs["traffic_signal"]),
            station=int(current_inputs["station"]),
            risk_dict=risk_dict,
            feature_order=feature_order,
            weather_categories=weather_categories,
        )
        rows.append(
            {
                "visibility": round(float(visibility), 2),
                "probability": predict_probability(model, frame),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_county_risk_df(csv_path: str, top_n: int = 12) -> pd.DataFrame:
    county_counts: dict[str, int] = {}
    for chunk in pd.read_csv(csv_path, usecols=["State", "County"], chunksize=250_000):
        ca_chunk = chunk.loc[chunk["State"] == "CA", ["County"]].dropna()
        counts = ca_chunk["County"].value_counts()
        for county, count in counts.items():
            county_counts[county] = county_counts.get(county, 0) + int(count)

    county_df = (
        pd.DataFrame({"county": list(county_counts.keys()), "incidents": list(county_counts.values())})
        .sort_values("incidents", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    county_df["log_incidents"] = np.log1p(county_df["incidents"])
    return county_df


def build_weather_comparison_df(
    model,
    risk_dict: dict[str, float],
    feature_order: list[str],
    weather_categories: list[str],
    current_inputs: dict[str, float | int | str],
) -> pd.DataFrame:
    preferred_weather = ["Clear", "Cloudy", "Fog", "Rain"]
    selected_weather = [w for w in preferred_weather if w in weather_categories]
    if not selected_weather:
        selected_weather = weather_categories[:4]

    rows = []
    for weather in selected_weather:
        frame, _, _ = make_scenario(
            lat=float(current_inputs["lat"]),
            lng=float(current_inputs["lng"]),
            hour=int(current_inputs["hour"]),
            temp=float(current_inputs["temp"]),
            hum=float(current_inputs["hum"]),
            vis=float(current_inputs["vis"]),
            wind=float(current_inputs["wind"]),
            weather_sel=str(weather),
            light_sel=str(current_inputs["light_sel"]),
            crossing=int(current_inputs["crossing"]),
            junction=int(current_inputs["junction"]),
            stop=int(current_inputs["stop"]),
            traffic_signal=int(current_inputs["traffic_signal"]),
            station=int(current_inputs["station"]),
            risk_dict=risk_dict,
            feature_order=feature_order,
            weather_categories=weather_categories,
        )
        rows.append({"weather": weather, "probability": predict_probability(model, frame)})
    return pd.DataFrame(rows)


def build_road_context_impact_df(
    model,
    risk_dict: dict[str, float],
    feature_order: list[str],
    weather_categories: list[str],
    current_inputs: dict[str, float | int | str],
) -> pd.DataFrame:
    contexts = [
        ("Baseline", 0, 0, 0, 0, 0),
        ("Crossing", 1, 0, 0, 0, 0),
        ("Junction", 0, 1, 0, 0, 0),
        ("Stop", 0, 0, 1, 0, 0),
        ("Traffic Signal", 0, 0, 0, 1, 0),
        ("Station", 0, 0, 0, 0, 1),
        ("All On", 1, 1, 1, 1, 1),
    ]
    rows = []
    for label, crossing, junction, stop, traffic_signal, station in contexts:
        frame, _, _ = make_scenario(
            lat=float(current_inputs["lat"]),
            lng=float(current_inputs["lng"]),
            hour=int(current_inputs["hour"]),
            temp=float(current_inputs["temp"]),
            hum=float(current_inputs["hum"]),
            vis=float(current_inputs["vis"]),
            wind=float(current_inputs["wind"]),
            weather_sel=str(current_inputs["weather_sel"]),
            light_sel=str(current_inputs["light_sel"]),
            crossing=crossing,
            junction=junction,
            stop=stop,
            traffic_signal=traffic_signal,
            station=station,
            risk_dict=risk_dict,
            feature_order=feature_order,
            weather_categories=weather_categories,
        )
        rows.append({"context": label, "probability": predict_probability(model, frame)})
    return pd.DataFrame(rows)


def build_feature_importance_df(model) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": list(model.feature_name_), "importance": list(model.feature_importances_)})
        .sort_values("importance", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def build_data_summary(csv_path: str, risk_dict: dict[str, float]) -> dict[str, int]:
    total_rows = 0
    ca_rows = 0
    county_counts: dict[str, int] = {}
    for chunk in pd.read_csv(csv_path, usecols=["State", "County"], chunksize=500_000):
        total_rows += len(chunk)
        ca_chunk = chunk.loc[chunk["State"] == "CA", ["County"]].dropna()
        ca_rows += len(ca_chunk)
        counts = ca_chunk["County"].value_counts()
        for county, count in counts.items():
            county_counts[county] = county_counts.get(county, 0) + int(count)

    top_county = max(county_counts.items(), key=lambda item: item[1])[0] if county_counts else "Unknown"
    return {
        "total_rows": int(total_rows),
        "ca_rows": int(ca_rows),
        "h3_cells": int(len(risk_dict)),
        "top_county": top_county,
    }


def build_generic_sensitivity_df(
    model,
    risk_dict: dict[str, float],
    feature_order: list[str],
    weather_categories: list[str],
    current_inputs: dict[str, float | int | str],
    variable: str,
    values: list[float],
) -> pd.DataFrame:
    rows = []
    for value in values:
        frame, _, _ = make_scenario(
            lat=float(current_inputs["lat"]),
            lng=float(current_inputs["lng"]),
            hour=int(current_inputs["hour"]),
            temp=float(current_inputs["temp"]) if variable != "temp" else float(value),
            hum=float(current_inputs["hum"]) if variable != "hum" else float(value),
            vis=float(current_inputs["vis"]) if variable != "vis" else float(value),
            wind=float(current_inputs["wind"]) if variable != "wind" else float(value),
            weather_sel=str(current_inputs["weather_sel"]),
            light_sel=str(current_inputs["light_sel"]),
            crossing=int(current_inputs["crossing"]),
            junction=int(current_inputs["junction"]),
            stop=int(current_inputs["stop"]),
            traffic_signal=int(current_inputs["traffic_signal"]),
            station=int(current_inputs["station"]),
            risk_dict=risk_dict,
            feature_order=feature_order,
            weather_categories=weather_categories,
        )
        rows.append({"value": float(value), "probability": predict_probability(model, frame)})
    return pd.DataFrame(rows)


def build_scenario_delta_df(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    baseline = benchmark_df.loc[benchmark_df["scenario"] == "Clear Midday Baseline", "probability"].iloc[0]
    current = benchmark_df.loc[benchmark_df["scenario"] == "Current Scenario", "probability"].iloc[0]
    la = benchmark_df.loc[benchmark_df["scenario"] == "LA Rain Commute", "probability"].iloc[0]
    return pd.DataFrame(
        {
            "comparison": ["Current vs Baseline", "Current vs LA Rain Commute"],
            "delta": [current - baseline, current - la],
        }
    )


def build_feature_snapshot_df(current_inputs: dict[str, float | int | str], current_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["Temperature", "Humidity", "Visibility", "Wind", "Neighbor Risk (log)"],
            "value": [
                float(current_inputs["temp"]),
                float(current_inputs["hum"]),
                float(current_inputs["vis"]),
                float(current_inputs["wind"]),
                float(current_df["neighbor_risk"].iloc[0]),
            ],
        }
    )


def build_h3_distribution_df(map_df: pd.DataFrame) -> pd.DataFrame:
    distribution = map_df.copy()
    max_incidents = int(distribution["incidents"].max())
    candidate_bins = [0, 100, 500, 1000, 2000, 5000, max_incidents + 1]
    candidate_labels = ["1-100", "101-500", "501-1000", "1001-2000", "2001-5000", "5000+"]

    bins: list[int] = [candidate_bins[0]]
    labels: list[str] = []
    for idx in range(1, len(candidate_bins)):
        if candidate_bins[idx] > bins[-1]:
            bins.append(candidate_bins[idx])
            labels.append(candidate_labels[idx - 1])

    if len(bins) < 2:
        bins = [0, max_incidents + 1]
        labels = [f"1-{max_incidents}"]

    distribution["incident_band"] = pd.cut(
        distribution["incidents"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    return distribution


def render_map(hex_map_df: pd.DataFrame, current_lat: float, current_lng: float) -> None:
    st.subheader("California Spatial Risk Map")
    st.caption("High-incident H3 neighborhoods from the training artifact, with your current scenario pinned on the map.")

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=hex_map_df,
            get_position="[lng, lat]",
            get_radius="incidents * 3",
            radius_min_pixels=3,
            radius_max_pixels=28,
            get_fill_color="[color_r, color_g, color_b, color_a]",
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"lat": current_lat, "lng": current_lng}]),
            get_position="[lng, lat]",
            get_radius=18000,
            radius_min_pixels=10,
            get_fill_color=[20, 33, 43, 220],
            pickable=True,
        ),
    ]

    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=36.7783, longitude=-119.4179, zoom=5, pitch=0),
        layers=layers,
        tooltip={"text": "H3: {hex_id}\nIncidents: {incidents}"},
    )
    st.pydeck_chart(deck, use_container_width=True)


def render_visuals(
    model,
    risk_dict: dict[str, float],
    feature_order: list[str],
    weather_categories: list[str],
    current_inputs: dict[str, float | int | str],
) -> None:
    current_df, current_hex_id, current_neighbor_risk = make_scenario(
        lat=float(current_inputs["lat"]),
        lng=float(current_inputs["lng"]),
        hour=int(current_inputs["hour"]),
        temp=float(current_inputs["temp"]),
        hum=float(current_inputs["hum"]),
        vis=float(current_inputs["vis"]),
        wind=float(current_inputs["wind"]),
        weather_sel=str(current_inputs["weather_sel"]),
        light_sel=str(current_inputs["light_sel"]),
        crossing=int(current_inputs["crossing"]),
        junction=int(current_inputs["junction"]),
        stop=int(current_inputs["stop"]),
        traffic_signal=int(current_inputs["traffic_signal"]),
        station=int(current_inputs["station"]),
        risk_dict=risk_dict,
        feature_order=feature_order,
        weather_categories=weather_categories,
    )
    current_probability = predict_probability(model, current_df)
    map_df = build_hex_map_df(risk_dict)
    benchmark_df = build_benchmark_df(model, risk_dict, feature_order, weather_categories, current_inputs)
    heatmap_df = build_weather_hour_heatmap_df(model, risk_dict, feature_order, weather_categories, current_inputs)
    sensitivity_df = build_visibility_sensitivity_df(model, risk_dict, feature_order, weather_categories, current_inputs)
    county_df = build_county_risk_df(str(PROJECT_ROOT / "data" / "US_Accidents.csv"))
    weather_df = build_weather_comparison_df(model, risk_dict, feature_order, weather_categories, current_inputs)
    importance_df = build_feature_importance_df(model)
    summary = build_data_summary(str(PROJECT_ROOT / "data" / "US_Accidents.csv"), risk_dict)
    humidity_df = build_generic_sensitivity_df(
        model, risk_dict, feature_order, weather_categories, current_inputs, "hum", list(np.linspace(20, 100, 9))
    )
    wind_df = build_generic_sensitivity_df(
        model, risk_dict, feature_order, weather_categories, current_inputs, "wind", list(np.linspace(0, 40, 9))
    )
    delta_df = build_scenario_delta_df(benchmark_df)
    snapshot_df = build_feature_snapshot_df(current_inputs, current_df)
    study_area = STUDY_AREAS[str(current_inputs["study_area"])]

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Accident Probability", f"{current_probability:.3%}")
    top2.metric("Current H3 Cell", current_hex_id)
    top3.metric("Neighbor Incidents", f"{current_neighbor_risk:,.0f}")
    top4.metric("Selected Weather", str(current_inputs["weather_sel"]))

    sum1, sum2, sum3, sum4 = st.columns(4)
    sum1.metric("US Rows", f"{summary['total_rows']:,}")
    sum2.metric("California Rows", f"{summary['ca_rows']:,}")
    sum3.metric("Tracked H3 Cells", f"{summary['h3_cells']:,}")
    sum4.metric("Top County", summary["top_county"])


    st.subheader("1. Study Area Lens")
    st.write(
        f"This dashboard can support a report section focused on **{current_inputs['study_area']}**. "
        f"For this area, the analysis emphasizes {study_area['focus']}. "
        "Use the map, hourly trend, benchmark comparison, and validation visuals below as report evidence."
    )

    scope_left, scope_right = st.columns([1, 1])
    with scope_left:
        st.subheader("California County Risk Ranking")
        county_chart = (
            alt.Chart(county_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=GOLD)
            .encode(
                x=alt.X("incidents:Q", title="Accident Count"),
                y=alt.Y("county:N", sort="-x", title="County"),
                tooltip=[
                    alt.Tooltip("county:N", title="County"),
                    alt.Tooltip("incidents:Q", title="Incidents"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(county_chart, use_container_width=True)

    with scope_right:
        st.subheader("Current Scenario Feature Snapshot")
        snapshot_chart = (
            alt.Chart(snapshot_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("value:Q", title="Current Value"),
                y=alt.Y("feature:N", sort="-x", title=None),
                color=alt.Color(
                    "feature:N",
                    scale=alt.Scale(
                        domain=["Temperature", "Humidity", "Visibility", "Wind", "Neighbor Risk (log)"],
                        range=[INK, ACCENT, GOLD, GREEN, DEEP],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("value:Q", title="Value", format=".3f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(snapshot_chart, use_container_width=True)

    st.subheader("2. Spatial Story")
    render_map(map_df, float(current_inputs["lat"]), float(current_inputs["lng"]))

    st.subheader("3. Time and Weather Story")
    top_left, top_right = st.columns([1.25, 1])
    with top_left:
        st.subheader("Risk by Hour and Weather")
        heatmap_chart = (
            alt.Chart(heatmap_df)
            .mark_rect(cornerRadius=3)
            .encode(
                x=alt.X("hour:O", title="Hour of Day"),
                y=alt.Y("weather:N", title="Weather"),
                color=alt.Color(
                    "probability:Q",
                    title="Probability",
                    scale=alt.Scale(range=["#F7D8CF", "#F09B83", ACCENT, DEEP]),
                ),
                tooltip=[
                    alt.Tooltip("weather:N", title="Weather"),
                    alt.Tooltip("hour:Q", title="Hour"),
                    alt.Tooltip("probability:Q", title="Probability", format=".3%"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(heatmap_chart, use_container_width=True)

    with top_right:
        st.subheader("Weather Condition Comparison")
        weather_chart = (
            alt.Chart(weather_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("weather:N", title="Weather"),
                y=alt.Y("probability:Q", title="Predicted Probability", axis=alt.Axis(format=".0%")),
                color=alt.Color(
                    "weather:N",
                    scale=alt.Scale(domain=list(weather_df["weather"]), range=[INK, ACCENT, GOLD, GREEN][: len(weather_df)]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("weather:N", title="Weather"),
                    alt.Tooltip("probability:Q", title="Probability", format=".3%"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(weather_chart, use_container_width=True)

    sensitivity_row_1, sensitivity_row_2, sensitivity_row_3 = st.columns([1, 1, 1])
    with sensitivity_row_1:
        st.subheader("Visibility Sensitivity")
        sensitivity_chart = (
            alt.Chart(sensitivity_df)
            .mark_line(point=alt.OverlayMarkDef(color=INK, filled=True, size=55), color=ACCENT, strokeWidth=3)
            .encode(
                x=alt.X("visibility:Q", title="Visibility (mi)"),
                y=alt.Y("probability:Q", title="Predicted Probability", axis=alt.Axis(format=".0%")),
                tooltip=[
                    alt.Tooltip("visibility:Q", title="Visibility", format=".2f"),
                    alt.Tooltip("probability:Q", title="Probability", format=".3%"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(sensitivity_chart, use_container_width=True)

    with sensitivity_row_2:
        st.subheader("Humidity Sensitivity")
        humidity_chart = (
            alt.Chart(humidity_df)
            .mark_line(point=alt.OverlayMarkDef(color=INK, filled=True, size=55), color=GOLD, strokeWidth=3)
            .encode(
                x=alt.X("value:Q", title="Humidity (%)"),
                y=alt.Y("probability:Q", title="Predicted Probability", axis=alt.Axis(format=".0%")),
                tooltip=[
                    alt.Tooltip("value:Q", title="Humidity", format=".1f"),
                    alt.Tooltip("probability:Q", title="Probability", format=".3%"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(humidity_chart, use_container_width=True)

    with sensitivity_row_3:
        st.subheader("Wind Speed Sensitivity")
        wind_chart = (
            alt.Chart(wind_df)
            .mark_line(point=alt.OverlayMarkDef(color=INK, filled=True, size=55), color=GREEN, strokeWidth=3)
            .encode(
                x=alt.X("value:Q", title="Wind Speed (mph)"),
                y=alt.Y("probability:Q", title="Predicted Probability", axis=alt.Axis(format=".0%")),
                tooltip=[
                    alt.Tooltip("value:Q", title="Wind Speed", format=".1f"),
                    alt.Tooltip("probability:Q", title="Probability", format=".3%"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(wind_chart, use_container_width=True)

    st.subheader("4. Scenario Story")
    delta_left, delta_right = st.columns([1, 1])
    with delta_left:
        st.subheader("Scenario Delta Comparison")
        delta_chart = (
            alt.Chart(delta_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("comparison:N", title=None),
                y=alt.Y("delta:Q", title="Probability Delta", axis=alt.Axis(format=".0%")),
                color=alt.condition(alt.datum.delta > 0, alt.value(ACCENT), alt.value(INK)),
                tooltip=[
                    alt.Tooltip("comparison:N", title="Comparison"),
                    alt.Tooltip("delta:Q", title="Delta", format=".3%"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(delta_chart, use_container_width=True)

    with delta_right:
        st.subheader("Interpretation Notes")
        st.write(
            f"In **{current_inputs['study_area']}**, the selected scenario currently scores **{current_probability:.3%}**. "
            f"Spatially, the map and top-H3 views show whether this point sits inside a historically dense neighborhood. "
            f"Temporally and environmentally, the heatmap plus sensitivity charts show how changing visibility, humidity, wind, and weather shifts the prediction. "
            f"Finally, the scenario delta chart shows how the current case compares with stronger and weaker reference conditions."
        )

    st.subheader("5. Model Understanding and Validation")
    lower_left, lower_right = st.columns([1, 1])
    with lower_left:
        st.subheader("Model Feature Importance")
        importance_chart = (
            alt.Chart(importance_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=DEEP)
            .encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("importance:Q", title="Importance"),
                ],
            )
            .properties(height=340)
        )
        st.altair_chart(importance_chart, use_container_width=True)

    with lower_right:
        st.subheader("Limitations and Notes")
        st.info(
            "This dashboard is designed for exploratory analysis and report support. "
            "The model uses California-only data, an out-of-time split, and synthetic-safe sampling. "
            "Some scenario rankings still need refinement, so interpretation should be framed as analytical rather than operational."
        )
        st.write(

 )

    st.subheader("Validation Visuals ")
    val1, val2, val3 = st.columns(3)
    with val1:
        st.image(str(PROJECT_ROOT / "performance_curves.png"), caption="ROC and Precision-Recall performance")
    with val2:
        st.image(str(PROJECT_ROOT / "confusion_matrix.png"), caption="Confusion matrix")
    with val3:
        st.image(str(PROJECT_ROOT / "cyclical_risk.png"), caption="Cyclical temporal risk behavior")

    st.subheader("Model Input Audit")
    st.dataframe(current_df, use_container_width=True)


def render_prediction_screen(
    model,
    risk_dict: dict[str, float],
    feature_order: list[str],
    weather_categories: list[str],
    current_inputs: dict[str, float | int | str],
) -> None:
    current_df, current_hex_id, current_neighbor_risk = make_scenario(
        lat=float(current_inputs["lat"]),
        lng=float(current_inputs["lng"]),
        hour=int(current_inputs["hour"]),
        temp=float(current_inputs["temp"]),
        hum=float(current_inputs["hum"]),
        vis=float(current_inputs["vis"]),
        wind=float(current_inputs["wind"]),
        weather_sel=str(current_inputs["weather_sel"]),
        light_sel=str(current_inputs["light_sel"]),
        crossing=int(current_inputs["crossing"]),
        junction=int(current_inputs["junction"]),
        stop=int(current_inputs["stop"]),
        traffic_signal=int(current_inputs["traffic_signal"]),
        station=int(current_inputs["station"]),
        risk_dict=risk_dict,
        feature_order=feature_order,
        weather_categories=weather_categories,
    )
    probability = predict_probability(model, current_df)

    st.subheader("Accident Probability")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Probability", f"{probability:.3%}")
    c2.metric("Current H3 Cell", current_hex_id)
    c3.metric("Neighbor Incidents", f"{current_neighbor_risk:,.0f}")

    if probability > 0.75:
        st.error("Critical risk. Conditions are strongly aligned with historical accident patterns.")
    elif probability > 0.45:
        st.warning("Elevated risk. Several conditions are pushing the score upward.")
    else:
        st.success("Stable scenario. The current pattern is lower risk relative to past incidents.")

    st.subheader("Scenario Summary")
    st.write(
        f"For **{current_inputs['study_area']}**, this scenario uses **{current_inputs['weather_sel']}** conditions at "
        f"**{current_inputs['hour']}:00** with visibility at **{current_inputs['vis']} mi** and wind speed at "
        f"**{current_inputs['wind']} mph**."
    )

    st.subheader("Model Input Audit")
    st.dataframe(current_df, use_container_width=True)


def main() -> None:
    validate_assets()
    model, risk_dict, feature_order, weather_categories = load_assets()

    st.title("California Real-Time Traffic Risk Engine")
    st.caption("Streamlit dashboard backed by our trained LightGBM classifier and H3 neighborhood risk features.")

    with st.sidebar:
        study_area_name = st.selectbox("Report Study Area", list(STUDY_AREAS.keys()), index=0)
        study_defaults = STUDY_AREAS[study_area_name]
        st.header("Location")
        lat = st.number_input("Latitude", value=float(study_defaults["lat"]), format="%.4f")
        lng = st.number_input("Longitude", value=float(study_defaults["lng"]), format="%.4f")
        hour = st.slider("Hour of Day (24h)", 0, 23, 17)
        light_sel = st.selectbox("Light Condition", ["Day", "Night"])

        st.header("Weather")
        temp = st.slider("Temperature (F)", 0, 120, 65)
        hum = st.slider("Humidity (%)", 0, 100, 90)
        vis = st.slider("Visibility (mi)", 0.0, 10.0, 2.0)
        wind = st.slider("Wind Speed (mph)", 0, 80, 20)
        default_weather = weather_categories.index("Rain") if "Rain" in weather_categories else 0
        weather_sel = st.selectbox("Weather Condition", weather_categories, index=default_weather)

        crossing = 0
        junction = 0
        stop = 0
        traffic_signal = 0
        station = 0

    current_inputs = {
        "lat": lat,
        "lng": lng,
        "study_area": study_area_name,
        "hour": hour,
        "temp": temp,
        "hum": hum,
        "vis": vis,
        "wind": wind,
        "weather_sel": weather_sel,
        "light_sel": light_sel,
        "crossing": int(crossing),
        "junction": int(junction),
        "stop": int(stop),
        "traffic_signal": int(traffic_signal),
        "station": int(station),
    }

    prediction_tab, analytics_tab = st.tabs(["Accident Probability", "Analytics & Visuals"])

    with prediction_tab:
        render_prediction_screen(model, risk_dict, feature_order, weather_categories, current_inputs)

    with analytics_tab:
        render_visuals(model, risk_dict, feature_order, weather_categories, current_inputs)

    with st.expander("View Feature Schema"):
        st.write(feature_order)


if __name__ == "__main__":
    main()
