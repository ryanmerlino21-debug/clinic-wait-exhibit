"""Train and evaluate a small neural network on the clinic wait-time data.

Split: 70% train / 15% validation / 15% test (fixed seed).
Hyperparameters are chosen on validation only. The test set is scored once at the end.
Outputs go to site/model.json (weights, scaler, metrics, held-out predictions).
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

SEED = 42
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

df = pd.read_csv("data/clinic-wait-times.csv")
print(df.day_of_week.unique())
DAYS = [d for d in DAYS if d in set(df.day_of_week)]

def features(d):
    X = pd.DataFrame({"hour": d.hour, "staff_on_shift": d.staff_on_shift,
                      "patients_in_queue": d.patients_in_queue})
    for day in DAYS:
        X["day_" + day] = (d.day_of_week == day).astype(float)
    return X

X = features(df)
y = df.wait_minutes.values
idx = np.arange(len(df))
tr, rest = train_test_split(idx, test_size=0.30, random_state=SEED)
va, te = train_test_split(rest, test_size=0.50, random_state=SEED)

mu, sd = X.iloc[tr].mean(), X.iloc[tr].std()
Z = ((X - mu) / sd).values

# Baseline: median wait of the training set
baseline = float(np.median(y[tr]))
base_mae = mean_absolute_error(y[te], np.full(len(te), baseline))

# Tune on validation only
best = None
for hidden in [(8,), (16,), (32,), (32, 16), (64, 32)]:
    for alpha in [1e-4, 1e-2]:
        m = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=3000,
                         random_state=SEED, early_stopping=False)
        m.fit(Z[tr], y[tr])
        mae = mean_absolute_error(y[va], m.predict(Z[va]))
        print(f"hidden={hidden} alpha={alpha} val MAE={mae:.2f}")
        if best is None or mae < best[0]:
            best = (mae, hidden, alpha, m)
val_mae, hidden, alpha, model = best
print("chosen:", hidden, alpha, "val MAE", round(val_mae, 2))

# Score the test set once
pred = model.predict(Z[te])
test_mae = mean_absolute_error(y[te], pred)
print(f"TEST MAE model={test_mae:.2f}  baseline={base_mae:.2f}")

# Held-out predictions for the gallery
test_rows = df.iloc[te].copy()
test_rows["predicted"] = np.round(pred, 1)
test_rows["error"] = np.round(pred - test_rows.wait_minutes, 1)
rows = test_rows.to_dict("records")

worst = max(rows, key=lambda r: abs(r["error"]))
print("worst miss:", worst)

# Where does it do worst? Error by staff level
test_rows["abs_err"] = test_rows.error.abs()
by_staff = test_rows.groupby("staff_on_shift").abs_err.mean().round(1).to_dict()
print("MAE by staff:", by_staff)

layers = []
for W, b in zip(model.coefs_, model.intercepts_):
    layers.append({"W": np.round(W, 6).tolist(), "b": np.round(b, 6).tolist()})

out = {
    "features": list(X.columns),
    "days": DAYS,
    "mean": mu.round(6).tolist(),
    "std": sd.round(6).tolist(),
    "layers": layers,
    "hidden": list(hidden),
    "alpha": alpha,
    "n_train": len(tr), "n_val": len(va), "n_test": len(te),
    "baseline_wait": round(baseline, 1),
    "metrics": {"model_mae": round(test_mae, 2), "baseline_mae": round(base_mae, 2),
                "val_mae": round(val_mae, 2),
                "model_r2": round(float(model.score(Z[te], y[te])), 3)},
    "mae_by_staff": {str(k): v for k, v in by_staff.items()},
    "worst": worst,
    "test_rows": rows,
    "wait_range": [float(y.min()), float(y.max())],
}
import os
os.makedirs("site", exist_ok=True)
json.dump(out, open("site/model.json", "w"), separators=(",", ":"))
print("wrote site/model.json")
