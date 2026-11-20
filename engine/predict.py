#!/usr/bin/env python3
"""Edge Predictor engine.

Downloads results, fixtures and bookmaker odds from football-data.co.uk,
rates every team, prices every upcoming match and writes app/data/predictions.json.

Model
  1. Team ratings: weighted Poisson (Maher / Dixon-Coles style) attack & defence
     ratings with home advantage, fitted per league by iterative proportional
     fitting. Recent matches weigh more (exponential time decay). Where the data
     has expected goals (xG), the target is a goals/xG mix, which is less noisy.
     Ratings are shrunk toward average so teams with few games don't go extreme.
  2. Score matrix: Poisson scorelines 0-10 with the Dixon-Coles low-score fix.
  3. Market blend: bookmaker odds are de-vigged (exchange prices preferred) and
     blended with the model. The expected goals are then re-solved so one
     consistent score matrix reproduces the blended 1X2 and Over/Under 2.5.
     Every other market (double chance, GG/NG, O/U 1.5/3.5, correct score) comes
     from that same matrix.
  4. Picks: per match, the safest pick (highest probability at fair odds >= 1.25)
     and a value pick (positive edge against the average or best available odds).

Usage
  python3 engine/predict.py                 # normal run, fixtures from today on
  python3 engine/predict.py --from 2026-11-14 --backtest
Standard library only.
"""
import argparse, csv, io, json, math, os, sys, time, urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "engine", ".cache")
OUT = os.path.join(ROOT, "app", "data", "predictions.json")
BASE = "https://www.football-data.co.uk/"
SEASON, PREV = "2627", "2526"

MAIN = {
    "E0": ("England", "Premier League"), "E1": ("England", "Championship"), "E2": ("England", "League One"),
    "E3": ("England", "League Two"), "EC": ("England", "National League"),
    "SC0": ("Scotland", "Premiership"), "SC1": ("Scotland", "Championship"), "SC2": ("Scotland", "League One"),
    "SC3": ("Scotland", "League Two"), "D1": ("Germany", "Bundesliga"), "D2": ("Germany", "2. Bundesliga"),
    "I1": ("Italy", "Serie A"), "I2": ("Italy", "Serie B"), "SP1": ("Spain", "La Liga"), "SP2": ("Spain", "Segunda"),
    "F1": ("France", "Ligue 1"), "F2": ("France", "Ligue 2"), "N1": ("Netherlands", "Eredivisie"),
    "B1": ("Belgium", "Pro League"), "P1": ("Portugal", "Primeira Liga"), "T1": ("Turkey", "Super Lig"),
    "G1": ("Greece", "Super League"),
}
EXTRA = {"Argentina": "ARG", "Austria": "AUT", "Brazil": "BRA", "China": "CHN", "Denmark": "DNK",
         "Finland": "FIN", "Ireland": "IRL", "Japan": "JPN", "Mexico": "MEX", "Norway": "NOR",
         "Poland": "POL", "Romania": "ROU", "Russia": "RUS", "Sweden": "SWE", "Switzerland": "SWZ", "USA": "USA"}

XI = 0.0025          # time decay per day (half-life ~ 9 months)
PRIOR = 3.0          # shrinkage strength, in "average matches"
RHO = -0.06          # Dixon-Coles low-score correlation
W_MKT_1X2, W_MKT_OU = 0.85, 0.75   # weight on market in the blend (backtest: market odds beat the pure model)
MAXG = 10
SAFE_MIN, SAFE_MAX = 1.25, 4.0   # fair-odds window for the 'safe' pick (below 1.25 SportyBet pays too little)


# ---------------------------------------------------------------- data
def fetch(path, max_age=3 * 3600):
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, path.replace("/", "_"))
    if os.path.exists(fn) and time.time() - os.path.getmtime(fn) < max_age:
        raw = open(fn, "rb").read()
    else:
        raw = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(BASE + path, headers={"User-Agent": "edge-predictor/1.0"})
                raw = urllib.request.urlopen(req, timeout=60).read()
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    print(f"  ! {path}: {e}", file=sys.stderr)
                    return []
                time.sleep(2 * (attempt + 1))
        open(fn, "wb").write(raw)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return [r for r in csv.DictReader(io.StringIO(text)) if any(r.values())]


def pdate(s):
    s = (s or "").strip()
    for f in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, f).date()
        except ValueError:
            pass
    return None


def fnum(r, *keys):
    for k in keys:
        v = (r.get(k) or "").strip()
        if v:
            try:
                x = float(v)
                if x > 0:
                    return x
            except ValueError:
                pass
    return None


def load_results(div):
    """Results for one league: list of dicts with date, home, away, hg, ag, hxg, axg, closing odds."""
    out = []
    if div in MAIN:
        rows = fetch(f"mmz4281/{PREV}/{div}.csv", 30 * 86400) + fetch(f"mmz4281/{SEASON}/{div}.csv")
        for r in rows:
            d, hg, ag = pdate(r.get("Date")), fnum(r, "FTHG"), fnum(r, "FTAG")
            if d is None or (r.get("FTHG") or "").strip() == "":
                continue
            out.append(dict(date=d, home=r["HomeTeam"].strip(), away=r["AwayTeam"].strip(),
                            hg=float(r["FTHG"]), ag=float(r["FTAG"]),
                            hxg=fnum(r, "HxG"), axg=fnum(r, "AxG"),
                            o=(fnum(r, "BFECH", "AvgCH"), fnum(r, "BFECD", "AvgCD"), fnum(r, "BFECA", "AvgCA")),
                            ou=(fnum(r, "BFEC>2.5", "AvgC>2.5"), fnum(r, "BFEC<2.5", "AvgC<2.5"))))
    else:
        cur = date.today().year
        for r in fetch(f"new/{div}.csv"):
            d = pdate(r.get("Date"))
            if d is None or (r.get("HG") or "").strip() == "" or d.year < cur - 2:
                continue
            out.append(dict(date=d, home=r["Home"].strip(), away=r["Away"].strip(),
                            hg=float(r["HG"]), ag=float(r["AG"]), hxg=None, axg=None,
                            o=(fnum(r, "PSCH", "AvgCH"), fnum(r, "PSCD", "AvgCD"), fnum(r, "PSCA", "AvgCA")),
                            ou=(None, None)))
    out.sort(key=lambda m: m["date"])
    return out


def load_fixtures():
    fx = []
    for r in fetch("fixtures.csv"):
        div = (r.get("Div") or "").strip()
        if div not in MAIN:
            continue
        c, l = MAIN[div]
        fx.append(dict(div=div, country=c, league=l, date=pdate(r.get("Date")), time=(r.get("Time") or "").strip(),
                       home=r["HomeTeam"].strip(), away=r["AwayTeam"].strip(),
                       sharp=(fnum(r, "BFEH"), fnum(r, "BFED"), fnum(r, "BFEA")),
                       avg=(fnum(r, "AvgH"), fnum(r, "AvgD"), fnum(r, "AvgA")),
                       best=(fnum(r, "MaxH"), fnum(r, "MaxD"), fnum(r, "MaxA")),
                       sharp_ou=(fnum(r, "BFE>2.5"), fnum(r, "BFE<2.5")),
                       avg_ou=(fnum(r, "Avg>2.5"), fnum(r, "Avg<2.5")),
                       best_ou=(fnum(r, "Max>2.5"), fnum(r, "Max<2.5"))))
    for r in fetch("new_league_fixtures.csv"):
        code = EXTRA.get((r.get("Country") or "").strip())
        if not code:
            continue
        fx.append(dict(div=code, country=r["Country"].strip(), league=r["League"].strip(), date=pdate(r.get("Date")),
                       time=(r.get("Time") or "").strip(), home=r["Home"].strip(), away=r["Away"].strip(),
                       sharp=(fnum(r, "PSH", "BFEH"), fnum(r, "PSD", "BFED"), fnum(r, "PSA", "BFEA")),
                       avg=(fnum(r, "AvgH"), fnum(r, "AvgD"), fnum(r, "AvgA")),
                       best=(fnum(r, "MaxH"), fnum(r, "MaxD"), fnum(r, "MaxA")),
                       sharp_ou=(None, None), avg_ou=(None, None), best_ou=(None, None)))
    return [f for f in fx if f["date"]]


# ---------------------------------------------------------------- model
class Ratings:
    def __init__(self, matches, ref):
        rows = []
        for m in matches:
            if m["date"] >= ref:
                continue
            w = math.exp(-XI * (ref - m["date"]).days)
            th = m["hg"] if m["hxg"] is None else 0.5 * m["hg"] + 0.5 * m["hxg"]
            ta = m["ag"] if m["axg"] is None else 0.5 * m["ag"] + 0.5 * m["axg"]
            rows.append((m["home"], m["away"], th, ta, w))
        self.rows = rows
        teams = {t for r in rows for t in r[:2]}
        self.att = {t: 1.0 for t in teams}
        self.dfn = {t: 1.0 for t in teams}
        self.n = defaultdict(float)
        for h, a, _, _, w in rows:
            self.n[h] += w; self.n[a] += w
        sw = sum(r[4] for r in rows) or 1
        self.mu = sum(r[3] * r[4] for r in rows) / sw or 1.1      # away goals per game
        self.hfa = (sum(r[2] * r[4] for r in rows) / sw) / self.mu if rows else 1.25
        self._fit()

    def _fit(self):
        if not self.rows:
            return
        for _ in range(40):
            sc, ec, cc, ecc = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
            for h, a, th, ta, w in self.rows:
                lh = self.mu * self.hfa * self.dfn[a]       # expected / att[h]
                la = self.mu * self.dfn[h]                  # expected / att[a]
                sc[h] += w * th; ec[h] += w * lh
                sc[a] += w * ta; ec[a] += w * la
                cc[a] += w * th; ecc[a] += w * self.mu * self.hfa * self.att[h]
                cc[h] += w * ta; ecc[h] += w * self.mu * self.att[a]
            g = (self.mu * (1 + self.hfa)) / 2
            for t in self.att:
                self.att[t] = (sc[t] + PRIOR * g) / (ec[t] + PRIOR * g)
            for t in self.dfn:
                self.dfn[t] = (cc[t] + PRIOR * g) / (ecc[t] + PRIOR * g)
            ga = math.exp(sum(math.log(v) for v in self.att.values()) / len(self.att))
            gd = math.exp(sum(math.log(v) for v in self.dfn.values()) / len(self.dfn))
            for t in self.att:
                self.att[t] /= ga; self.dfn[t] /= gd
            num_h = sum(w * th for h, a, th, ta, w in self.rows)
            den_h = sum(w * self.mu * self.att[h] * self.dfn[a] for h, a, th, ta, w in self.rows)
            num_a = sum(w * ta for h, a, th, ta, w in self.rows)
            den_a = sum(w * self.att[a] * self.dfn[h] for h, a, th, ta, w in self.rows)
            self.mu = num_a / den_a
            self.hfa = num_h / den_h

    def xg(self, h, a):
        ah, dh = self.att.get(h, 0.92), self.dfn.get(h, 1.08)   # unknown team: assume slightly weak (promoted)
        aa, da = self.att.get(a, 0.92), self.dfn.get(a, 1.08)
        return self.mu * self.hfa * ah * da, self.mu * aa * dh

    def games(self, t):
        return round(self.n.get(t, 0), 1)


def pois(k, l):
    return math.exp(-l + k * math.log(l) - math.lgamma(k + 1)) if l > 0 else (1.0 if k == 0 else 0.0)


def matrix(lh, la, rho=RHO):
    ph = [pois(i, lh) for i in range(MAXG + 1)]
    pa = [pois(j, la) for j in range(MAXG + 1)]
    m = [[ph[i] * pa[j] for j in range(MAXG + 1)] for i in range(MAXG + 1)]
    m[0][0] *= max(0.0, 1 - lh * la * rho); m[0][1] *= max(0.0, 1 + lh * rho)
    m[1][0] *= max(0.0, 1 + la * rho); m[1][1] *= max(0.0, 1 - rho)
    s = sum(map(sum, m))
    return [[p / s for p in row] for row in m]


def summarize(m):
    H = D = A = GG = 0.0
    ou = {l: 0.0 for l in (0.5, 1.5, 2.5, 3.5, 4.5)}
    h05 = a05 = h15 = a15 = 0.0
    for i, row in enumerate(m):
        for j, p in enumerate(row):
            if i > j: H += p
            elif i == j: D += p
            else: A += p
            if i and j: GG += p
            for l in ou:
                if i + j > l: ou[l] += p
            if i > 0: h05 += p
            if j > 0: a05 += p
            if i > 1: h15 += p
            if j > 1: a15 += p
    return dict(H=H, D=D, A=A, GG=GG, ou=ou, h05=h05, a05=a05, h15=h15, a15=a15)


def devig(odds):
    if not odds or any(o is None or o <= 1 for o in odds):
        return None
    inv = [1 / o for o in odds]
    s = sum(inv)
    return [i / s for i in inv]


def solve_xg(target, lh0, la0):
    """Find expected goals whose score matrix best matches the target 1X2 (+ O2.5)."""
    def loss(lh, la):
        s = summarize(matrix(lh, la))
        e = (s["H"] - target[0]) ** 2 + (s["D"] - target[1]) ** 2 + (s["A"] - target[2]) ** 2
        if target[3] is not None:
            e += (s["ou"][2.5] - target[3]) ** 2
        return e
    best = (lh0, la0, loss(lh0, la0))
    for step in (0.3, 0.15, 0.08, 0.04, 0.02, 0.01):
        improved = True
        while improved:
            improved = False
            for dh, da in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                lh, la = best[0] + dh * step, best[1] + da * step
                if 0.05 < lh < 6 and 0.05 < la < 6:
                    e = loss(lh, la)
                    if e < best[2] - 1e-12:
                        best, improved = (lh, la, e), True
    return best[0], best[1]


def price(rt, home, away, x1x2, xou):
    """Model + market -> final expected goals and summary. x1x2/xou: (odds tuples) or None."""
    mh, ma = rt.xg(home, away)
    model = summarize(matrix(mh, ma))
    fair = devig(x1x2) if x1x2 else None
    fou = devig(xou) if xou else None
    if fair:
        t = [W_MKT_1X2 * fair[k] + (1 - W_MKT_1X2) * p for k, p in enumerate((model["H"], model["D"], model["A"]))]
        to = W_MKT_OU * fou[0] + (1 - W_MKT_OU) * model["ou"][2.5] if fou else None
        lh, la = solve_xg(t + [to], mh, ma)
    else:
        lh, la = mh, ma
    return dict(model_xg=(mh, ma), xg=(lh, la), model=model, fair=fair, fair_ou=fou)


# ---------------------------------------------------------------- picks
def markets(s, home, away):
    ou = s["ou"]
    return [
        ("1X2", "1", f"{home} to win", s["H"]), ("1X2", "X", "Draw", s["D"]), ("1X2", "2", f"{away} to win", s["A"]),
        ("Double Chance", "1X", f"{home} or draw", s["H"] + s["D"]),
        ("Double Chance", "X2", f"{away} or draw", s["D"] + s["A"]),
        ("Double Chance", "12", "No draw", s["H"] + s["A"]),
        ("Over/Under", "Over 1.5", "2+ goals", ou[1.5]), ("Over/Under", "Under 1.5", "0-1 goals", 1 - ou[1.5]),
        ("Over/Under", "Over 2.5", "3+ goals", ou[2.5]), ("Over/Under", "Under 2.5", "0-2 goals", 1 - ou[2.5]),
        ("Over/Under", "Over 3.5", "4+ goals", ou[3.5]), ("Over/Under", "Under 3.5", "0-3 goals", 1 - ou[3.5]),
        ("GG/NG", "GG", "Both teams score", s["GG"]), ("GG/NG", "NG", "Not both score", 1 - s["GG"]),
        ("Home Over/Under", "Home Over 0.5", f"{home} scores", s["h05"]),
        ("Away Over/Under", "Away Over 0.5", f"{away} scores", s["a05"]),
        ("Home Over/Under", "Home Over 1.5", f"{home} 2+ goals", s["h15"]),
        ("Away Over/Under", "Away Over 1.5", f"{away} 2+ goals", s["a15"]),
    ]


def build(fx, rt):
    has_odds = bool(devig(fx["sharp"]) or devig(fx["avg"]))
    x1x2 = fx["sharp"] if devig(fx["sharp"]) else fx["avg"] if devig(fx["avg"]) else None
    xou = fx["sharp_ou"] if devig(fx["sharp_ou"]) else fx["avg_ou"] if devig(fx["avg_ou"]) else None
    pr = price(rt, fx["home"], fx["away"], x1x2, xou)
    m = matrix(*pr["xg"])
    s = summarize(m)
    odds_for = {"1": (fx["avg"][0], fx["best"][0]), "X": (fx["avg"][1], fx["best"][1]), "2": (fx["avg"][2], fx["best"][2]),
                "Over 2.5": (fx["avg_ou"][0], fx["best_ou"][0]), "Under 2.5": (fx["avg_ou"][1], fx["best_ou"][1])}
    mk = []
    for market, code, label, p in markets(s, fx["home"], fx["away"]):
        avg, best = odds_for.get(code, (None, None))
        mk.append(dict(market=market, pick=code, label=label, p=round(p, 4), fair=round(1 / max(p, 1e-6), 2),
                       avg=avg, best=best,
                       edge_avg=round(p * avg - 1, 4) if avg else None,
                       edge_best=round(p * best - 1, 4) if best else None))
    playable = [x for x in mk if SAFE_MIN <= x["fair"] <= SAFE_MAX]
    safe = max(playable, key=lambda x: x["p"])
    value_c = [x for x in mk if x["edge_best"] is not None and x["edge_best"] > 0.02 and x["p"] >= 0.25]
    value = max(value_c, key=lambda x: (x["edge_avg"] or -1, x["edge_best"])) if value_c else None

    gh, ga = rt.games(fx["home"]), rt.games(fx["away"])
    data_q = min(1.0, min(gh, ga) / 10)
    if pr["fair"]:
        dm = pr["model"]
        agree = 1 - min(1.0, 2.5 * max(abs(dm["H"] - pr["fair"][0]), abs(dm["A"] - pr["fair"][2])))
    else:
        agree = 0.4
    conf = round(100 * (0.65 * safe["p"] + 0.2 * agree + 0.15 * data_q))
    cs = sorted(((f"{i}-{j}", m[i][j]) for i in range(7) for j in range(7)), key=lambda x: -x[1])[:6]
    return dict(
        id=f"{fx['div']}-{fx['date'].isoformat()}-{fx['home']}-{fx['away']}".replace(" ", "_"),
        country=fx["country"], league=fx["league"], date=fx["date"].isoformat(), time=fx["time"],
        home=fx["home"], away=fx["away"],
        xg=[round(pr["xg"][0], 2), round(pr["xg"][1], 2)], model_xg=[round(v, 2) for v in pr["model_xg"]],
        p1x2=[round(s["H"], 4), round(s["D"], 4), round(s["A"], 4)],
        model_1x2=[round(pr["model"][k], 4) for k in ("H", "D", "A")],
        market_1x2=[round(v, 4) for v in pr["fair"]] if pr["fair"] else None,
        has_odds=has_odds, games=[gh, ga], confidence=conf, agreement=round(agree, 2),
        safe=safe, value=value, markets=mk, scores=[[c, round(p, 4)] for c, p in cs],
    )


# ---------------------------------------------------------------- backtest
def backtest(results_by_div, since):
    """Walk forward over this season: refit weekly on data before each week, then score that week's matches."""
    rows = []
    for div, res in results_by_div.items():
        if div not in MAIN:
            continue
        tests = [m for m in res if m["date"] >= since]
        weeks = defaultdict(list)
        for m in tests:
            weeks[m["date"] - timedelta(days=m["date"].weekday())].append(m)
        for wk, ms in sorted(weeks.items()):
            rt = Ratings(res, wk)
            for m in ms:
                pr = price(rt, m["home"], m["away"], m["o"] if devig(m["o"]) else None, m["ou"] if devig(m["ou"]) else None)
                s = summarize(matrix(*pr["xg"]))
                res_idx = 0 if m["hg"] > m["ag"] else 1 if m["hg"] == m["ag"] else 2
                mk = markets(s, "H", "A")
                fair = [1 / max(x[3], 1e-6) for x in mk]
                cand = [(x, f) for x, f in zip(mk, fair) if SAFE_MIN <= f <= SAFE_MAX]
                safe = max(cand, key=lambda c: c[0][3])[0]
                rows.append(dict(div=div, res=res_idx, hg=m["hg"], ag=m["ag"],
                                 model=[pr["model"][k] for k in ("H", "D", "A")], final=[s["H"], s["D"], s["A"]],
                                 market=pr["fair"], safe=safe[1], safe_p=safe[3], safe_hit=hit(safe[1], m["hg"], m["ag"])))
    if not rows:
        return None

    def ll(key):
        xs = [r for r in rows if r[key]] if key == "market" else rows
        return round(-sum(math.log(max(r[key][r["res"]], 1e-9)) for r in xs) / len(xs), 4), len(xs)

    def acc(key):
        xs = [r for r in rows if r[key]]
        return round(sum(1 for r in xs if max(range(3), key=lambda k: r[key][k]) == r["res"]) / len(xs), 4)

    buckets = []
    for lo, hi in ((0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9)):
        b = [r for r in rows if lo <= r["safe_p"] < hi]
        if b:
            buckets.append(dict(range=f"{int(lo*100)}-{int(hi*100)}%", n=len(b),
                                predicted=round(sum(r["safe_p"] for r in b) / len(b), 4),
                                actual=round(sum(r["safe_hit"] for r in b) / len(b), 4)))
    top = sorted(rows, key=lambda r: -r["safe_p"])[: max(1, len(rows) // 5)]
    return dict(since=since.isoformat(), matches=len(rows),
                logloss=dict(model=ll("model")[0], market=ll("market")[0], blended=ll("final")[0]),
                accuracy=dict(model=acc("model"), market=acc("market"), blended=acc("final")),
                safe_hit=round(sum(r["safe_hit"] for r in rows) / len(rows), 4),
                safe_pred=round(sum(r["safe_p"] for r in rows) / len(rows), 4),
                top20_hit=round(sum(r["safe_hit"] for r in top) / len(top), 4),
                top20_pred=round(sum(r["safe_p"] for r in top) / len(top), 4),
                calibration=buckets)


def hit(code, hg, ag):
    t = hg + ag
    return {
        "1": hg > ag, "X": hg == ag, "2": hg < ag, "1X": hg >= ag, "X2": hg <= ag, "12": hg != ag,
        "Over 1.5": t > 1.5, "Under 1.5": t < 1.5, "Over 2.5": t > 2.5, "Under 2.5": t < 2.5,
        "Over 3.5": t > 3.5, "Under 3.5": t < 3.5, "GG": hg > 0 and ag > 0, "NG": not (hg > 0 and ag > 0),
        "Home Over 0.5": hg > 0, "Away Over 0.5": ag > 0, "Home Over 1.5": hg > 1, "Away Over 1.5": ag > 1,
    }[code]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", help="first fixture date YYYY-MM-DD (default: today, UTC)")
    ap.add_argument("--backtest", action="store_true", help="also run the walk-forward backtest")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    start = date.fromisoformat(a.start) if a.start else datetime.now(timezone.utc).date()

    fixtures = [f for f in load_fixtures() if f["date"] >= start]
    divs = sorted({f["div"] for f in fixtures} | (set(MAIN) if a.backtest else set()))
    print(f"{len(fixtures)} fixtures from {start} in {len({f['div'] for f in fixtures})} leagues")
    results, ratings = {}, {}
    for d in divs:
        results[d] = load_results(d)
        ratings[d] = Ratings(results[d], start)
        print(f"  {d}: {len(results[d])} results, {len(ratings[d].att)} teams")

    out = []
    for f in sorted(fixtures, key=lambda f: (f["date"], f["time"], f["country"], f["league"])):
        try:
            out.append(build(f, ratings[f["div"]]))
        except Exception as e:  # noqa: BLE001
            print(f"  ! {f['home']} v {f['away']}: {e}", file=sys.stderr)

    # last round: picks made before kickoff, graded against the real results
    recent = []
    for f in sorted((f for f in load_fixtures() if start - timedelta(days=10) <= f["date"] < start),
                    key=lambda f: (f["date"], f["time"])):
        res = results.get(f["div"]) or load_results(f["div"])
        results.setdefault(f["div"], res)
        r = next((m for m in res if m["date"] == f["date"] and m["home"] == f["home"] and m["away"] == f["away"]), None)
        if not r:
            continue
        try:
            b = build(f, Ratings(res, f["date"]))
        except Exception:  # noqa: BLE001
            continue
        recent.append(dict(id=b["id"], country=b["country"], league=b["league"], date=b["date"], home=b["home"],
                           away=b["away"], score=f"{int(r['hg'])}-{int(r['ag'])}", pick=b["safe"]["pick"],
                           label=b["safe"]["label"], p=b["safe"]["p"], confidence=b["confidence"],
                           won=hit(b["safe"]["pick"], r["hg"], r["ag"])))
    print(f"last round: {sum(x['won'] for x in recent)}/{len(recent)} safe picks won")

    prev = None
    if os.path.exists(a.out):
        try:
            prev = json.load(open(a.out)).get("backtest")
        except Exception:  # noqa: BLE001
            prev = None
    bt = backtest(results, date(int(SEASON[:2]) + 2000, 9, 15)) if a.backtest else prev

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(generated=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), start=start.isoformat(),
                   source="football-data.co.uk", matches=out, recent=recent, backtest=bt),
              open(a.out, "w"), separators=(",", ":"))
    print(f"wrote {len(out)} matches -> {os.path.relpath(a.out, ROOT)}")
    if bt:
        print(json.dumps(bt, indent=1))


if __name__ == "__main__":
    main()
