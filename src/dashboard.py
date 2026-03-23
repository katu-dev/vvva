import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from simulator import F1Simulator
from predictor import F1Predictor

st.set_page_config(
    page_title="VVVA F1 Predictor",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Fond principal */
    .stApp { background-color: #0e0e0e; color: #f0f0f0; }

    /* Header custom */
    .f1-header {
        background: linear-gradient(135deg, #e10600 0%, #8b0000 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .f1-header h1 { color: white; margin: 0; font-size: 2rem; font-weight: 900; letter-spacing: 2px; }
    .f1-header p  { color: rgba(255,255,255,0.75); margin: 0; font-size: 0.9rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: #1a1a1a; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; color: #aaa; font-weight: 600; padding: 8px 20px; }
    .stTabs [aria-selected="true"] { background: #e10600 !important; color: white !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 1rem;
    }
    [data-testid="stMetricValue"] { color: #e10600; font-size: 2rem; font-weight: 900; }
    [data-testid="stMetricLabel"] { color: #aaa; }

    /* Bouton */
    .stButton > button {
        background: linear-gradient(135deg, #e10600, #ff4444);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 0.6rem 2rem;
        transition: all 0.2s;
    }
    .stButton > button:hover { transform: scale(1.03); box-shadow: 0 4px 20px rgba(225,6,0,0.4); }

    /* Selectbox & inputs */
    .stSelectbox > div > div, .stNumberInput > div > div > input {
        background: #1a1a1a !important;
        border: 1px solid #333 !important;
        color: #f0f0f0 !important;
        border-radius: 8px !important;
    }

    /* Dataframe */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #111; border-right: 1px solid #222; }

    /* Podium couleurs */
    .pos-1 { color: #FFD700; font-weight: 900; }
    .pos-2 { color: #C0C0C0; font-weight: 800; }
    .pos-3 { color: #CD7F32; font-weight: 700; }

    /* Section title */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e10600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        border-left: 3px solid #e10600;
        padding-left: 10px;
        margin: 1.2rem 0 0.8rem 0;
    }
    h2, h3 { color: #f0f0f0 !important; }

    /* Success */
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="f1-header">
    <div>
        <h1>VVVA - FORMULA 1 RACE SIMULATOR</h1>
        <p>Simulateur & Prédicteur basé sur 15 ans de données réelles (2009 – 2024)</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Cache ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_simulator():
    return F1Simulator()

@st.cache_resource
def load_predictor():
    predictor = F1Predictor()
    with st.spinner("Entraînement du modèle ML..."):
        scores = predictor.train()
    return predictor, scores

simulator   = load_simulator()
circuits_df = simulator.get_available_circuits()

WEATHER_LABELS = {"sunny": "Ensoleillé", "cloudy": "Nuageux", "rain": "Pluie"}
PLOTLY_THEME   = "plotly_dark"

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Simulateur", "Prédicteur ML", "Statistiques"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Simulateur
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Paramètres de la course</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_circuit = st.selectbox(
            "Circuit",
            circuits_df['circuitId'].values,
            format_func=lambda x: circuits_df[circuits_df['circuitId'] == x]['name'].values[0]
        )
    with col2:
        weather_key = st.selectbox("Météo", list(WEATHER_LABELS.keys()),
                                   format_func=lambda x: WEATHER_LABELS[x])
    with col3:
        year = st.number_input("Année", min_value=2009, max_value=2024, value=2024)

    circuit_info = circuits_df[circuits_df['circuitId'] == selected_circuit].iloc[0]
    st.caption(f"{circuit_info['name']} — {circuit_info['location']}, {circuit_info['country']}")

    st.markdown("")
    if st.button("Lancer la simulation", type="primary"):
        with st.spinner("Simulation en cours..."):
            results = simulator.simulate_race(circuit_id=selected_circuit, weather=weather_key, year=year)

        st.success(f"Course simulée ! {len(results)} pilotes classés.")

        # ── Podium ──
        st.markdown('<p class="section-title">Podium</p>', unsafe_allow_html=True)
        medals = ["🥇", "🥈", "🥉"]
        p_cols = st.columns(3)
        for i, col in enumerate(p_cols):
            row = results.iloc[i]
            with col:
                st.markdown(f"""
                <div style="background:#1a1a1a;border-radius:12px;padding:1.2rem;text-align:center;
                            border-top: 3px solid {'#FFD700' if i==0 else '#C0C0C0' if i==1 else '#CD7F32'}">
                    <div style="font-size:2.5rem">{medals[i]}</div>
                    <div style="font-size:1.2rem;font-weight:900;color:#f0f0f0">{row['driver']}</div>
                    <div style="color:#aaa;font-size:0.85rem">{row['team']}</div>
                    <div style="color:#e10600;font-weight:700;margin-top:4px">Score {row['performance']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # ── Tableau + Bar chart ──
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown('<p class="section-title">Classement complet</p>', unsafe_allow_html=True)
            st.dataframe(
                results.style.apply(
                    lambda row: [
                        "color: #FFD700; font-weight:900" if row['final_position'] == 1 else
                        "color: #C0C0C0; font-weight:800" if row['final_position'] == 2 else
                        "color: #CD7F32; font-weight:700" if row['final_position'] == 3 else ""
                    ] * len(row),
                    axis=1
                ),
                use_container_width=True,
                hide_index=True
            )

        with col_right:
            st.markdown('<p class="section-title">Performance Top 10</p>', unsafe_allow_html=True)
            fig = px.bar(
                results.head(10),
                x='code', y='performance', color='team',
                template=PLOTLY_THEME,
                title=f"Top 10 — {WEATHER_LABELS[weather_key]}",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig.update_layout(
                paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
                font_color='#f0f0f0', legend_title_text='Équipe',
                xaxis_title="Pilote", yaxis_title="Score"
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Line chart progression ──
        st.markdown('<p class="section-title">Progression Grille → Arrivée (Top 10)</p>', unsafe_allow_html=True)
        fig2 = go.Figure()
        colors = px.colors.qualitative.Bold
        for i, (_, row) in enumerate(results.head(10).iterrows()):
            fig2.add_trace(go.Scatter(
                x=['Départ', 'Arrivée'],
                y=[row['grid_position'], row['final_position']],
                mode='lines+markers+text',
                name=row['code'],
                text=[row['code'], ""],
                textposition="middle left",
                line=dict(width=2.5, color=colors[i % len(colors)]),
                marker=dict(size=8)
            ))
        fig2.update_layout(
            template=PLOTLY_THEME,
            paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
            font_color='#f0f0f0',
            yaxis=dict(autorange='reversed', title='Position', gridcolor='#2a2a2a'),
            xaxis=dict(gridcolor='#2a2a2a'),
            height=420,
            legend_title_text='Pilote'
        )
        st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Prédicteur ML
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    predictor, scores = load_predictor()

    st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Train Score", f"{scores['train_score'] * 100:.1f}%")
    with col2:
        st.metric("Test Score", f"{scores['test_score'] * 100:.1f}%",
                  delta=f"{(scores['test_score'] - scores['train_score']) * 100:.1f}%")
    with col3:
        st.metric("Samples", f"{scores['n_samples']:,}")

    st.markdown("---")

    mode_col, circuit_col = st.columns([1, 2])
    with mode_col:
        mode = st.radio("Mode", ["Predict 2026", "Real past results"])
    with circuit_col:
        pred_circuit = st.selectbox(
            "Circuit",
            circuits_df['circuitId'].values,
            format_func=lambda x: circuits_df[circuits_df['circuitId'] == x]['name'].values[0],
            key="pred_circuit"
        )

    if mode == "Predict 2026":
        st.markdown('<p class="section-title">2026 Race Prediction</p>', unsafe_allow_html=True)
        st.caption("Based on 2024 driver lineup, trained on 2009–2024 historical data.")

        pred_results = predictor.predict_2026_race(circuit_id=pred_circuit)
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.dataframe(pred_results, use_container_width=True, hide_index=True)

        with col_right:
            fig = px.bar(
                pred_results.head(10), x='code', y='predicted_position',
                color='team', template=PLOTLY_THEME,
                title="Predicted Top 10 — 2026",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig.update_layout(
                paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
                font_color='#f0f0f0', yaxis=dict(autorange='reversed', title='Predicted Position'),
                xaxis_title="Driver"
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        available_years = predictor.data_loader.get_available_years_for_circuit(pred_circuit)
        if not available_years:
            st.warning("No historical data for this circuit.")
        else:
            selected_year = st.selectbox("Year", available_years)
            real = predictor.get_real_results(circuit_id=pred_circuit, year=selected_year)

            if real.empty:
                st.warning("No results found for this race.")
            else:
                st.markdown(f'<p class="section-title">Real Results — {selected_year}</p>', unsafe_allow_html=True)
                col_left, col_right = st.columns([1, 1])

                with col_left:
                    st.dataframe(real, use_container_width=True, hide_index=True)

                with col_right:
                    fig = px.bar(
                        real.head(10), x='code', y='points',
                        color='team', template=PLOTLY_THEME,
                        title=f"Points scored — Top 10 ({selected_year})",
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    fig.update_layout(
                        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
                        font_color='#f0f0f0', xaxis_title="Driver", yaxis_title="Points"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-title">Feature Importance</p>', unsafe_allow_html=True)
    importance = predictor.get_feature_importance()
    fig_imp = px.bar(
        importance, x='importance', y='feature',
        orientation='h', template=PLOTLY_THEME,
        color='importance',
        color_continuous_scale=[[0, '#8b0000'], [1, '#e10600']],
        title="What factors influence the prediction most?"
    )
    fig_imp.update_layout(
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font_color='#f0f0f0', showlegend=False,
        coloraxis_showscale=False,
        yaxis_title="", xaxis_title="Importance"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Statistiques
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Circuits disponibles</p>', unsafe_allow_html=True)

    col_search, _ = st.columns([1, 2])
    with col_search:
        search = st.text_input("Rechercher un circuit", placeholder="Monaco, Spa...")

    filtered = circuits_df
    if search:
        mask = circuits_df.apply(lambda r: search.lower() in str(r).lower(), axis=1)
        filtered = circuits_df[mask]

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.caption(f"{len(filtered)} circuit(s) affiché(s)")

    st.markdown('<p class="section-title">Répartition par pays</p>', unsafe_allow_html=True)
    country_counts = circuits_df['country'].value_counts().reset_index()
    country_counts.columns = ['Pays', 'Circuits']
    fig3 = px.bar(
        country_counts.head(15), x='Pays', y='Circuits',
        template=PLOTLY_THEME,
        color='Circuits',
        color_continuous_scale=[[0, '#8b0000'], [1, '#e10600']],
        title="Top 15 pays avec le plus de circuits F1"
    )
    fig3.update_layout(
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font_color='#f0f0f0', coloraxis_showscale=False
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0">
        <div style="font-size:3rem"><svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" id="F1--Streamline-Simple-Icons" height="48" width="48"><desc>F1 Streamline Icon: https://streamlinehq.com</desc><title>F1</title><path d="M9.6 11.24h7.91L19.75 9H9.39c-2.85 0 -3.62 0.34 -5.17 1.81C2.71 12.3 0 15 0 15h3.38c0.77 -0.75 2.2 -2.13 2.85 -2.75 0.92 -0.87 1.37 -1.01 3.37 -1.01zM20.39 9l-6 6H18l6 -6h-3.61zm-3.25 2.61H9.88c-2.22 0 -2.6 0.12 -3.55 1.07C5.44 13.57 4 15 4 15h3.15l0.75 -0.75c0.49 -0.49 0.75 -0.55 1.78 -0.55h5.37l2.09 -2.09z" fill="#ff0004" stroke-width="1"></path></svg></div>
        <div style="font-size:1.3rem;font-weight:900;color:#e10600;letter-spacing:2px">VVVA F1</div>
        <div style="color:#888;font-size:0.8rem">Race Prediction System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Données**")
    st.markdown(f"- {len(circuits_df)} circuits")
    st.markdown("- Saisons 2009 → 2024")
    st.markdown("- Pilotes, équipes, résultats")

    st.markdown("---")
    st.markdown("**Météo**")
    for key, label in WEATHER_LABELS.items():
        st.markdown(f"- {label}")

    st.markdown("---")
    st.caption("Projet VVVA · F1 Prediction")
