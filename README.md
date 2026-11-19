# Clinic Wait-Time Simulator

A small neural network predicts patient wait time in a simulated clinic. Visitors set the day, hour, staff on shift and queue length, compare staffing levels, find the smallest staff count that meets a target wait, and inspect predictions on 300 held-out visits. An optional in-browser chatbot (WebLLM, WebGPU) answers questions from a short guide.

- **Live site:** https://ryanmerlino21-debug.github.io/clinic-wait-exhibit/
- **Site source:** `site/` (static, no build step). Deployed by `.github/workflows/pages.yml`.
- **Training and evaluation script:** `train.py` (scikit-learn `MLPRegressor`). `python3 train.py` regenerates `site/model.json`.
- **Data:** `data/clinic-wait-times.csv`, from https://snsie.github.io/neural-networks/data/clinic-wait-times.csv. Synthetic: 2,000 simulated visits.

## Model and evaluation

| | Test MAE (min) |
|---|---|
| Baseline (training median, 60.3 min) | 44.27 |
| Neural network (8 inputs, 64 and 32 ReLU units, 1 output) | 4.94 |

70/15/15 train/validation/test split (seed 42). Architecture and L2 strength were chosen on validation MAE only; the test set was scored once. The network runs in the browser as a plain JavaScript forward pass over exported weights, checked against scikit-learn on all 300 test rows.

**Highlighted mistake:** Mon 11 AM, 5 staff, 15 waiting. Actual 37.4 min, predicted 55.7 min. Similar visits range from about 37 to 70 min, so this is mostly noise the inputs cannot explain. Only 22 of 2,000 visits fall in that corner.

## Exercise requirements checklist

| Requirement (from the exercise page) | Where |
|---|---|
| Train and evaluate a small network; define inputs and outputs | `train.py`; site section 4 |
| Keep the test set separate from training and tuning | 70/15/15 split; tuning on validation only |
| Compare against a simple baseline on the test set | Median baseline, site section 3 |
| Pick one mistake and explain what might have helped | Site sections 3 and 4 |
| One meaningful interaction with one clear discovery | Simulator and staffing planner (diminishing returns of added staff) |
| Held-out predictions can be explored | Scatter plot and sortable, downloadable table, site section 3 |
| Run the trained model on new inputs | Simulator runs the network in the browser |
| WebGPU chatbot with WebLLM | Site section 5 |
| Guide covering purpose, data, inputs, results, limitations | `site/guide.txt`, shown on the site |
| Chatbot says when the guide does not answer | System prompt in `site/app.js` |
| Check WebGPU; "Enable chat" button; loading progress; site works without chat | Site section 5 |
| Test three questions (Data, Results, Outside the guide) and check answers | `site/chatbot-tests.json`, shown in a table on the site (**not yet run**) |
| Credit the data source | Site section 6 and footer |
| Link to site, source, and training script | Footer and this README |
| Both partners can explain a prediction and a chatbot check | See the team section below |

## Still to do before submitting

1. **Run the chatbot checks.** Open the live site in a WebGPU browser, select Enable chat, ask the three questions, compare each answer with the site, and fill in `site/chatbot-tests.json` (`answer`, `supported`, optional `note`). Revise `site/guide.txt` if an answer is unsupported. The site table updates automatically.
2. **Answer the Canvas questions** (not visible in the exercise page).
3. **Add your partner** as a repository collaborator.

## Team

Roles from the exercise: one partner leads model and evaluation, the other leads the website and chatbot; both write the guide.

| Name | Role |
|---|---|
| _add name_ | Model and evaluation lead |
| _add name_ | Website and chatbot lead |

## Limits

Synthetic data only. The model describes the simulation, not a real clinic.

## Process note

Built with help from Claude Code (code, site and write-up drafts). Review the code and results yourself before submitting, and follow your course's AI-use policy.
