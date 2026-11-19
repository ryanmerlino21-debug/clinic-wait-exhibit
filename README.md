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
| Test three questions (Data, Results, Outside the guide) and check answers | Run on the live site; results and fixes in `site/chatbot-tests.json`, shown in a table on the site |
| Credit the data source | Site section 6 and footer |
| Link to site, source, and training script | Footer and this README |
| Both partners can explain a prediction and a chatbot check | See the team section below |

## Chatbot test results

Run on the live site with WebLLM (Llama-3.2-1B, WebGPU). The first run found an invented answer: asked how the model did on a dataset it was never tested on, the bot said it "performed poorly" with an error of 18.3 minutes. That number is the error on one single visit, and no other dataset was ever tested. Fixes: a "not covered" section in the guide, a stricter system prompt, an example refusal and a key-facts block. After the fixes all three answers match the site. Full before/after answers are in the table on the site.

## Features beyond the minimum

Staffing planner, hour-by-hour staffing schedule, wait map (staff by queue length), scenario comparison with CSV export, shareable scenario links, 90% prediction range, a you-versus-model challenge with running average error, sortable and downloadable held-out predictions, light/dark/auto theme, and saved comparison and score (stored only in your browser).

## Still to do before submitting

1. **Answer the Canvas questions** (not visible in the exercise page).
2. **Add your partner** as a repository collaborator and fill in the team table below.

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
