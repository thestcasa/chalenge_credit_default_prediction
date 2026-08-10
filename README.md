# Credit Card Default Prediction

Task: predict credit-card default risk for the DSML Lab evaluation set.

Final selected setup:
- Model family: CatBoost
- Setup: prune02
- Seed: 123
- Threshold: 0.328
- Public Macro F1: 0.721
- Predicted defaults: 1258

Data is expected under `data/`:
- `data/dev.csv`
- `data/eval.csv`
- `data/submission.csv`


Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

Expected output:
- `submission.csv` at the project root contains the selected public 0.721 run, 1st place in the course leaderboard

