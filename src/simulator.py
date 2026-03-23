import numpy as np
import pandas as pd
try:
    from data_loader import F1DataLoader        # streamlit (src/ in sys.path)
except ImportError:
    from src.data_loader import F1DataLoader    # programmatic usage from root


class F1Simulator:
    """
    Realistic F1 Grand Prix simulator.

    Performance model (weighted):
      40% — constructor strength  (the car is the dominant factor in F1)
      35% — recent driver form    (last 5 races)
      15% — circuit history       (driver's record on this track)
      10% — pure driver skill     (performance vs their car over career)

    Weather model:
      - speed multiplier applied to all drivers
      - in rain: car matters less, driver skill amplified (wet specialists shine)
      - DNF risk = constructor reliability × weather factor (varies per team)

    Circuit model:
      - overtaking difficulty computed from historical position changes
      - harder to overtake → grid position is more deterministic → less randomness
    """

    WEATHER = {
        'sunny': {'speed': 1.00, 'reliability': 0.95, 'skill_amplifier': 1.0},
        'cloudy': {'speed': 0.98, 'reliability': 0.93, 'skill_amplifier': 1.1},
        'rain':   {'speed': 0.85, 'reliability': 0.80, 'skill_amplifier': 1.8},
    }

    def __init__(self):
        self.data_loader = F1DataLoader()
        # Precompute once at startup
        self._driver_skill     = self.data_loader.get_driver_skill_vs_car()
        self._constructor_dnf  = self.data_loader.get_constructor_reliability()

    # ── Public API ───────────────────────────────────────────────────────────

    def simulate_race(self, circuit_id: int, weather: str, year: int = 2025) -> pd.DataFrame:
        """
        Simulate a full race.

        Parameters
        ----------
        circuit_id : int
        weather    : 'sunny' | 'cloudy' | 'rain'
        year       : season year (drivers lineup taken from 2024 if year > 2024)
        """
        w = self.WEATHER.get(weather, self.WEATHER['sunny'])
        drivers = self.data_loader.get_drivers_for_year(min(year, 2024))
        overtaking = self.data_loader.get_circuit_overtaking_factor(circuit_id)

        results = []
        for _, driver in drivers.iterrows():
            entry = self._simulate_driver(driver, circuit_id, year, w, overtaking, len(results) + 1)
            results.append(entry)

        df = pd.DataFrame(results)
        # DNF drivers sorted to the bottom
        df = df.sort_values(['dnf', 'performance'], ascending=[True, False]).reset_index(drop=True)
        df['final_position'] = df.index + 1
        df['status'] = df['dnf'].map({True: 'DNF', False: 'Classé'})

        return df[['final_position', 'driver', 'code', 'team', 'grid_position', 'performance', 'status']]

    def get_available_circuits(self):
        return self.data_loader.circuits[['circuitId', 'name', 'location', 'country']]

    # ── Internal ─────────────────────────────────────────────────────────────

    def _simulate_driver(self, driver, circuit_id, year, w, overtaking, grid_pos):
        did = driver['driverId']
        cid = driver['constructorId']

        # ── 1. Constructor strength (car performance, 0→1) ───────────────────
        car = self.data_loader.get_constructor_strength(cid, up_to_year=year)

        # ── 2. Recent driver form (0→1) ──────────────────────────────────────
        form = self.data_loader.get_recent_form(did, year=year)

        # ── 3. Circuit history (0→1) ─────────────────────────────────────────
        circuit_avg = self.data_loader.get_circuit_form(did, circuit_id)
        circuit_skill = 1 - (circuit_avg / 20)

        # ── 4. Pure driver skill vs their car (clipped to ±0.15) ────────────
        raw_skill = self._driver_skill.get(did, 0.0)
        skill = float(np.clip(raw_skill, -0.15, 0.15))

        # ── 5. Weighted base performance ─────────────────────────────────────
        base = (
            car          * 0.40 +
            form         * 0.35 +
            circuit_skill * 0.15 +
            (skill + 0.5) * 0.10   # shift to ~0.5 centre
        )

        # ── 6. Weather effect ────────────────────────────────────────────────
        # In rain the car advantage shrinks and driver skill is amplified
        if w['skill_amplifier'] > 1.0:
            # Reduce car weight, amplify skill component
            rain_base = (
                car          * 0.25 +   # car matters less in rain
                form         * 0.35 +
                circuit_skill * 0.15 +
                (skill + 0.5) * (0.10 * w['skill_amplifier'])  # skill amplified
            )
            # Blend between base and rain_base
            alpha = (w['skill_amplifier'] - 1.0)  # 0 for sunny, 0.8 for rain
            base = base * (1 - alpha * 0.3) + rain_base * (alpha * 0.3)

        performance = base * w['speed']

        # ── 7. Randomness — harder circuits = less randomness ────────────────
        # Monaco: overtaking=0.1 → std=0.04 (very deterministic)
        # Monza:  overtaking=1.0 → std=0.10 (lots of movement)
        noise_std = 0.04 + overtaking * 0.06
        performance += np.random.normal(0, noise_std)

        # ── 8. DNF — per-constructor reliability scaled by weather ───────────
        base_dnf = float(self._constructor_dnf.get(cid, 0.05))
        # Weather degrades reliability: rain = /0.80
        dnf_risk = base_dnf / w['reliability']
        dnf_risk = float(np.clip(dnf_risk, 0.01, 0.40))
        dnf = np.random.random() < dnf_risk

        return {
            'driver':        f"{driver['forename']} {driver['surname']}",
            'code':          driver['code'],
            'team':          driver['team'],
            'performance':   float(np.clip(performance, 0, None)),
            'grid_position': grid_pos,
            'dnf':           dnf,
        }
