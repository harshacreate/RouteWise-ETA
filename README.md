# RouteWise-ETA — Delivery-Time Forecasting Prototype

**Sector:** Logistics  
**Task:** Regression: delivery duration in minutes

Estimates delivery duration from distance, stops, congestion, weather, departure hour, route complexity, and urban share. Routes are grouped in the holdout.

## Data and split

The deterministic offline generator in `routewise/data.py` creates synthetic examples and uses no external dataset. Route IDs are held out as groups and kept separate from model features.

## Methods and metrics

- **Baseline:** training-mean predictor.
- **Stronger model:** histogram gradient-boosting regression.
- **Metrics:** MAE, RMSE, R², 90th-percentile absolute error, and coverage within 15 minutes.
- **Error analysis:** tail absolute error and tolerance coverage are emitted for both models.

## Limitations

Synthetic travel conditions are simplified; no live traffic data are used. The model cannot account for dispatch, local routing, customer access, or rapidly changing road conditions.

## Run

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -e ".[test]"
    python -m routewise.train --seed 42 --samples 2400 --json
    python -m pytest -q

`make setup`, `make train`, and `make test` are shortcuts. Python 3.9 or newer is required.

## License

MIT; see [LICENSE](LICENSE). No external data is included.
