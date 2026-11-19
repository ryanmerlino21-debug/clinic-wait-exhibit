# Clinic Wait-Time Simulator

A small neural network predicts patient wait time in a simulated clinic. Visitors change the hour, weekday, staff on shift and queue length, and compare the model's predictions on 300 held-out visits. An optional in-browser chatbot (WebLLM, WebGPU) answers questions from a short guide.

- **Site:** `site/index.html` (static, no build step). Deployed to GitHub Pages by `.github/workflows/pages.yml`.
- **Training script:** `train.py` (scikit-learn `MLPRegressor`). Run `python3 train.py` to regenerate `site/model.json`.
- **Data:** `data/clinic-wait-times.csv` from https://snsie.github.io/neural-networks/data/clinic-wait-times.csv. It is synthetic (2,000 simulated visits).

## Model and evaluation

| | Test MAE (min) |
|---|---|
| Baseline (training median, 60.3 min) | 44.27 |
| Neural network (64-32 ReLU MLP) | 4.94 |

70/15/15 train/validation/test split (seed 42). Architecture and L2 strength were chosen on validation MAE; the test set was scored once. Inputs are hour, staff, queue and weekday one-hot, standardized with training statistics. The network runs in the browser as a plain JavaScript forward pass over exported weights (checked against scikit-learn on all 300 test rows).

**Highlighted mistake:** Mon 11:00, 5 staff, 15 in queue: actual 37.4 min, predicted 55.7 min. Similar visits range from about 37 to 70 min, so this is mostly noise the inputs can't explain.

## Chatbot test log

`CHATBOT_TESTS.md` has the three required questions (Data, Results, Outside the guide). Fill in the real answers after running them in a WebGPU browser.

## Limits

Synthetic data only. The model describes the simulation, not a real clinic.

## Process note

Built with help from Claude Code (code, site and write-up drafts). Review the code and results yourself before submitting, and follow your course's AI-use policy.
