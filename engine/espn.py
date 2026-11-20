"""ESPN public scoreboard feed: daily fixtures (with DraftKings odds where offered) and results
for 150+ competitions. One request per day: /apis/site/v2/sports/soccer/all/scoreboard?dates=YYYYMMDD
"""
import json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache", "espn")
URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard?dates={}&limit=1000"
LEAGUES = json.load(open(os.path.join(HERE, "espn_leagues.json"), encoding="utf-8"))
SKIP = ("usa.ncaa",)  # college soccer: not offered by mainstream bookmakers


def _get(day, final):
    """Fetch one day. Days that are fully in the past are cached for good; others for 1 hour."""
    os.makedirs(CACHE, exist_ok=True)
    key = day.strftime("%Y%m%d")
    fn = os.path.join(CACHE, key + ".json")
    if os.path.exists(fn) and (final or time.time() - os.path.getmtime(fn) < 3600):
        try:
            return json.load(open(fn, encoding="utf-8"))
        except ValueError:
            pass
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL.format(key), headers={"User-Agent": "edge-predictor/1.0"})
            raw = urllib.request.urlopen(req, timeout=60).read()
            d = json.loads(raw.decode("utf-8"))
            json.dump(d, open(fn, "w", encoding="utf-8"))
            return d
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"  ! espn {key}: {e}", file=sys.stderr)
                return {"events": []}
            time.sleep(2 * (attempt + 1))


def american(s):
    try:
        v = float(str(s).replace("+", ""))
    except (TypeError, ValueError):
        return None
    if v == 0:
        return None
    return round(1 + (v / 100 if v > 0 else 100 / -v), 3)


def _league(e):
    m = re.search(r"l:(\d+)", e.get("uid", ""))
    lid = m.group(1) if m else "0"
    info = LEAGUES.get(lid)
    if info:
        return lid, info["slug"], info["name"]
    slug = (e.get("season") or {}).get("slug", "")
    name = re.sub(r"^\d{4}(-\d{2})?-", "", slug).replace("-", " ").title() or "Other"
    return lid, "", name


def _odds(comp):
    o = next((x for x in (comp.get("odds") or []) if x), {})
    ml = o.get("moneyline") or {}
    def close(side):
        s = ml.get(side) or {}
        return american((s.get("close") or s.get("open") or {}).get("odds"))
    x = (close("home"), close("draw"), close("away"))
    ou = (None, None)
    tot = o.get("total") or {}
    if o.get("overUnder") == 2.5 and tot:
        def tclose(side):
            s = tot.get(side) or {}
            c = s.get("close") or s.get("open") or {}
            return american(c.get("odds")) if str(c.get("line", "")).endswith("2.5") else None
        ou = (tclose("over"), tclose("under"))
    return (x if all(x) else (None, None, None)), (ou if all(ou) else (None, None))


def parse(day_json):
    out = []
    for e in day_json.get("events", []):
        try:
            comp = e["competitions"][0]
            lid, slug, lname = _league(e)
            if slug.startswith(SKIP):
                continue
            teams = {c["homeAway"]: c for c in comp["competitors"]}
            h, a = teams["home"], teams["away"]
            st = comp["status"]["type"]
            ko = datetime.strptime(e["date"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
            x, ou = _odds(comp)
            out.append(dict(
                eid=e["id"], lid=lid, slug=slug, league=lname, ko=ko,
                home=h["team"].get("displayName") or h["team"].get("name"),
                away=a["team"].get("displayName") or a["team"].get("name"),
                hk="espn:" + str(h["team"]["id"]), ak="espn:" + str(a["team"]["id"]),
                state=st.get("state"), completed=bool(st.get("completed")),
                hg=_int(h.get("score")), ag=_int(a.get("score")),
                neutral=bool(comp.get("neutralSite")), odds=x, ou=ou))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def load(history_days=160, ahead_days=2, now=None):
    """Return (results, fixtures). Results: completed matches in the last `history_days`.
    Fixtures: not-yet-started matches from now through `ahead_days` days ahead (UTC)."""
    now = now or datetime.now(timezone.utc)
    today = now.date()
    days = [today - timedelta(days=i) for i in range(history_days, 0, -1)] + \
           [today + timedelta(days=i) for i in range(0, ahead_days + 1)]
    with ThreadPoolExecutor(8) as ex:
        data = list(ex.map(lambda d: _get(d, d < today - timedelta(days=1)), days))
    seen, results, fixtures = set(), [], []
    for d in data:
        for m in parse(d):
            if m["eid"] in seen:
                continue
            seen.add(m["eid"])
            if m["completed"] and m["hg"] is not None:
                results.append(m)
            elif m["state"] == "pre" and m["ko"] > now:
                fixtures.append(m)
    results.sort(key=lambda m: m["ko"])
    fixtures.sort(key=lambda m: m["ko"])
    return results, fixtures
