# Edge Predictor

Automatic football predictions for 30+ leagues, for your phone.

- `engine/predict.py` downloads results, fixtures and odds from football-data.co.uk, rates every team, prices every upcoming match, grades last round's picks and runs a walk-forward backtest. Output: `app/data/predictions.json`.
- `app/index.html` is the phone app: match list with best picks and confidence, auto-pick of the best 2–20 games, slip with smart split and copy-to-SportyBet, and a track record view.
- `app/manual.html` is the original calculator for entering a match by hand.

Run: `python3 engine/predict.py --backtest` (standard library only).

## Original manual calculator

## What it does
- Turns recent goals scored and conceded into expected goals for each team, with small samples shrunk toward the league average.
- Builds a full scoreline matrix with a Dixon-Coles Poisson model, giving 1X2, Double Chance, Over/Under, GG/NG, team goals and correct score.
- Removes SportyBet's margin from the odds you enter and blends that market price with the model.
- Flags **value bets** (positive edge) and suggests a quarter-Kelly stake.
- Betslip builder that shows the true chance of an accumulator winning and copies your picks. SportyBet gives you the booking code when you add the picks and tap **Book Bet**.

## Use it on your phone
1. In GitHub, go to **Settings → Pages** and deploy from this branch, folder `/ (root)`.
2. Open `https://<your-username>.github.io/<repo>/app/` on your phone.
3. Use the browser menu and choose **Add to Home screen**.

Or open `app/index.html` in any browser.

## Honest note
No model can guarantee wins. This one helps you avoid bad-value bets and size your stakes sensibly. Bet only what you can afford to lose.
