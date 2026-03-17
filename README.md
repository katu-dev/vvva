# VVVA Project - F1 Race Results Prediction

## Description
Formula 1 race results prediction system based on real historical data (2009-2024) and weather influence. The project combines a Grand Prix simulator and a machine learning model to predict driver performances.

## Features
- GP simulator with real data (drivers, teams, circuits)
- ML model (Random Forest) trained on 15 years of F1 data
- Weather impact simulation (sunny, cloudy, rain)
- Interactive dashboard with Plotly visualizations
- Recent driver form analysis
- Performance history by circuit

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Dashboard (recommended)
```bash
python -m streamlit run src/dashboard.py
```

The dashboard offers 3 tabs:
1. **Simulator**: Run race simulations with different circuits and weather conditions
2. **ML Predictor**: View model performance and feature importance
3. **Statistics**: Explore circuit data

### Programmatic Usage

#### Simulator
```python
from src.simulator import F1Simulator

simulator = F1Simulator()
results = simulator.simulate_race(circuit_id=1, weather="rain", year=2024)
print(results)
```

#### Predictor
```python
from src.predictor import F1Predictor

predictor = F1Predictor()
scores = predictor.train()
print(f"Model score: {scores['test_score']:.3f}")
```

## Project Structure
```
vvva-f1/
├── csv/                    # F1 data (2009-2024)
│   ├── races.csv
│   ├── results.csv
│   ├── drivers.csv
│   ├── circuits.csv
│   └── constructors.csv
├── src/
│   ├── data_loader.py      # Data loading and preparation
│   ├── predictor.py        # ML prediction model
│   ├── simulator.py        # Grand Prix simulator
│   └── dashboard.py        # Streamlit interface
├── docs/
│   └── DOCUMENTATION.md    # Documentation complète en français
├── requirements.txt
└── README.md
```

## Documentation
La documentation complète en français (avec schémas et explications ligne par ligne) est disponible dans [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).

## Data
The data comes from the complete history of modern F1 and includes:
- Race results (2009-2024)
- Driver and team information
- Circuit characteristics
- Starting and finishing positions
- Lap times and points

## Technologies Used
- **Streamlit**: Interactive dashboard
- **Plotly**: Dynamic visualizations
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning (Random Forest)
- **NumPy**: Numerical computations

## Prediction Model
The Random Forest model uses the following features:
- Grid position (starting position)
- Circuit (encoded)
- Driver age
- Race year

Typical score: ~0.6-0.7 (R²)

## Authors
VVVA Project - F1 Prediction
