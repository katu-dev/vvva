import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from simulator import F1Simulator
from predictor import F1Predictor

st.set_page_config(page_title="VVVA F1 Predictor", layout="wide")

st.title("VVVA - Simulateur et Prédicteur F1")
st.caption("Basé sur données historiques réelles de F1")

# Initialisation
@st.cache_resource
def load_simulator():
    return F1Simulator()

@st.cache_resource
def load_predictor():
    predictor = F1Predictor()
    with st.spinner("Entraînement du modèle ML..."):
        scores = predictor.train()
    return predictor, scores

simulator = load_simulator()
circuits_df = simulator.get_available_circuits()

# Tabs
tab1, tab2, tab3 = st.tabs(["Simulateur", "Prédicteur ML", "Statistiques"])

with tab1:
    st.header("Simulateur de Grand Prix")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_circuit = st.selectbox(
            "Circuit",
            circuits_df['circuitId'].values,
            format_func=lambda x: circuits_df[circuits_df['circuitId']==x]['name'].values[0]
        )

    with col2:
        weather = st.selectbox("Météo", ["sunny", "cloudy", "rain"])

    with col3:
        year = st.number_input("Année", min_value=2009, max_value=2024, value=2024)

    if st.button("Lancer la simulation", type="primary"):
        with st.spinner("Simulation en cours..."):
            results = simulator.simulate_race(circuit_id=selected_circuit, weather=weather, year=year)

        st.success("Simulation terminée!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Résultats de la course")
            st.dataframe(results, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Performance par pilote")
            fig = px.bar(
                results.head(10),
                x='code',
                y='performance',
                color='team',
                title=f"Top 10 - {weather.capitalize()}"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Progression Grid → Final")
        fig2 = go.Figure()

        for _, row in results.head(10).iterrows():
            fig2.add_trace(go.Scatter(
                x=['Grid', 'Final'],
                y=[row['grid_position'], row['final_position']],
                mode='lines+markers',
                name=row['code'],
                line=dict(width=2)
            ))

        fig2.update_layout(
            title="Évolution des positions (Top 10)",
            yaxis=dict(autorange='reversed', title='Position'),
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header("Prédicteur Machine Learning")

    predictor, scores = load_predictor()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score Train", f"{scores['train_score']:.3f}")
    with col2:
        st.metric("Score Test", f"{scores['test_score']:.3f}")
    with col3:
        st.metric("Échantillons", scores['n_samples'])

    st.subheader("Importance des features")
    importance = predictor.get_feature_importance()
    fig = px.bar(importance, x='importance', y='feature', orientation='h')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Statistiques des circuits")
    st.dataframe(circuits_df, use_container_width=True, hide_index=True)

st.sidebar.markdown("---")
st.sidebar.info("""
**Projet VVVA**
Prédiction de résultats F1 avec influence météo
Données: 2009-2024
""")
