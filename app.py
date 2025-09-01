"""
Full VIP Ultra v3 - Sicbo predictor (no numpy)
- Ensemble (deterministic dropout), meta-learner, markov, freq, theory
- Entropy, autocorrelation, volatility implemented with math & statistics
- Save/load state to JSON
"""

from flask import Flask, jsonify, request, abort
import urllib.request
import json, math, os, statistics
from collections import defaultdict, Counter, deque
from time import time

app = Flask(__name__)

# ---------------- CONFIG -----------------
STATE_FILE = "sicbo_state.json"
API_URL = "https://sicbosunwin.onrender.com/api/sicbo/sunwin"  # remote source
MAX_HISTORY = 1000
ENSEMBLE_SIZE = 100
TOP_K_DEFAULT = 3
ADMIN_TOKEN = os.environ.get("SICBO_ADMIN_TOKEN", "letmein123")
LAPLACE_K = 1.0
L2_REG = 1e-4
RATE_LIMIT_WINDOW = 3.0
RATE_LIMIT_MAX = 10
EMA_ALPHA = 0.08
DROPOUT_RATE = 0.1  # deterministic dropout fraction
STATE_SAVE_INTERVAL = 1  # save every update

# ---------------- GLOBAL STATE -----------------
history = deque(maxlen=MAX_HISTORY)        # list of totals (3..18)
last_phien = None
trans = defaultdict(lambda: defaultdict(int))   # markov counts
ensemble = []   # list of model dicts {"id":int, "bias":..., "weights":{...}, "lr":..., "score_ema": float}
meta = {"weights": defaultdict(float), "bias": 0.0}  # stacking meta learner with weights
metrics = {"total_predictions": 0, "correct_predictions": 0, "rolling": deque(maxlen=200)}
rate_limits = defaultdict(list)
last_save_time = 0.0

# ---------------- THEORY PROB -----------------
def sicbo_probabilities():
    counts = {3:1,4:3,5:6,6:10,7:15,8:21,9:25,10:27,
              11:27,12:25,13:21,14:15,15:10,16:6,17:3,18:1}
    total = sum(counts.values())
    return {s: counts[s]/total for s in range(3,19)}
P_THEORY = sicbo_probabilities()

# ---------------- SAVE/LOAD STATE -----------------
def save_state():
    global last_save_time
    try:
        data = {
            "history": list(history),
            "last_phien": last_phien,
            "trans": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in trans.items()},
            "ensemble": ensemble,
            "meta": {"weights": dict(meta["weights"]), "bias": meta["bias"]},
            "metrics": {
                "total_predictions": metrics["total_predictions"],
                "correct_predictions": metrics["correct_predictions"],
                "rolling": list(metrics["rolling"])
            }
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        last_save_time = time()
    except Exception as e:
        print("Save state error:", e)

def load_state():
    global history, last_phien, trans, ensemble, meta, metrics
    if not os.path.exists(STATE_FILE):
        init_ensemble()
        return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        history.clear()
        for x in data.get("history", []):
            history.append(int(x))
        last_phien = data.get("last_phien")
        raw_trans = data.get("trans", {})
        trans.clear()
        for k, row in raw_trans.items():
            kk = int(k)
            for k2, v2 in row.items():
                trans[kk][int(k2)] = int(v2)
        loaded_ens = data.get("ensemble", [])
        # Basic validation: ensure it is a list and has at most ENSEMBLE_SIZE items
        if isinstance(loaded_ens, list) and len(loaded_ens) == ENSEMBLE_SIZE:
            ensemble.clear()
            for m in loaded_ens:
                # convert weights to normal dict if necessary
                m.setdefault("weights", {})
                m.setdefault("score_ema", 0.5)
                ensemble.append(m)
        else:
            init_ensemble()
        mdata = data.get("meta", {"weights": {}, "bias": 0.0})
        meta["weights"] = defaultdict(float, mdata.get("weights", {}))
        meta["bias"] = mdata.get("bias", 0.0)
        m = data.get("metrics", {})
        metrics["total_predictions"] = m.get("total_predictions", 0)
        metrics["correct_predictions"] = m.get("correct_predictions", 0)
        metrics["rolling"] = deque(m.get("rolling", []), maxlen=200)
    except Exception as e:
        print("Load state error:", e)
        init_ensemble()

# ---------------- INIT ENSEMBLE (DETERMINISTIC) -----------------
def init_ensemble():
    global ensemble
    ensemble = []
    for i in range(ENSEMBLE_SIZE):
        lr = 0.025 + (i % 25) * 0.0018   # spreads lr deterministically
        bias = -0.08 + ((i // 10) - 5) * 0.02
        ensemble.append({
            "id": i,
            "bias": float(bias),
            "weights": {},    # feature_name -> float
            "lr": float(lr),
            "score_ema": 0.5  # track recent accuracy/perf for weighting
        })

# ---------------- MARKOV & FREQUENCY -----------------
def update_markov(prev_sum, next_sum):
    if prev_sum is None or next_sum is None:
        return
    trans[prev_sum][next_sum] += 1

def markov_row_probs(last_sum):
    counts = trans.get(last_sum, {})
    total = sum(counts.values())
    denom = float(total) + LAPLACE_K * 16.0
    probs = {}
    for s in range(3, 19):
        probs[s] = (counts.get(s, 0) + LAPLACE_K) / denom
    return probs

def frequency_probs():
    if not history:
        return {s: 1.0/16.0 for s in range(3, 19)}
    c = Counter(history)
    total = float(sum(c.values()))
    return {s: (c.get(s, 0) + 0.0) / total for s in range(3, 19)}

# ---------------- FEATURES (NO NUMPY) -----------------
def normalize_sum(s):
    return (s - 3) / 15.0

def streaks_tai_xiu(hist):
    if not hist:
        return 0, 0
    last = hist[-1]
    is_tai = last >= 11
    streak = 0
    for x in reversed(hist):
        if (x >= 11) == is_tai:
            streak += 1
        else:
            break
    return (streak, 0) if is_tai else (0, streak)

def calculate_entropy(hist):
    if not hist:
        return 0.0
    c = Counter(hist)
    probs = [c.get(s, 0)/len(hist) for s in range(3, 19) if c.get(s, 0) > 0]
    if not probs:
        return 0.0
    ent = -sum(p * math.log(p) for p in probs)
    # normalize by log(16)
    return ent / math.log(16)

def autocorrelation(hist, lag=1):
    if len(hist) < lag + 1:
        return 0.0
    x = hist[:-lag]
    y = hist[lag:]
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y))
    return (num / den) if den > 0 else 0.0

def volatility(hist):
    if len(hist) < 2:
        return 0.0
    # population stddev normalized by max possible spread (~7.5)
    return statistics.pstdev(hist) / 7.5

def features_for_state(hist):
    feats = {}
    n = len(hist)
    feats["bias1"] = 1.0
    if n == 0:
        feats.update({
            "last_norm": 0.5,
            "delta_last": 0.0,
            "tai_ratio_5": 0.5,
            "tai_ratio_10": 0.5,
            "tai_ratio_20": 0.5,
            "streak_tai": 0.0,
            "streak_xiu": 0.0,
            "markov_tai_row": 0.5,
            "near_modal": 1.0,
            "entropy_20": 0.5,
            "autocorr_lag1": 0.0,
            "autocorr_lag2": 0.0,
            "volatility_10": 0.0
        })
        return feats

    last = hist[-1]
    feats["last_norm"] = normalize_sum(last)
    feats["delta_last"] = (hist[-1] - hist[-2]) / 15.0 if n >= 2 else 0.0
    for w in (5, 10, 20):
        sub = hist[-w:] if n >= w else hist
        feats[f"tai_ratio_{w}"] = (sum(1 for s in sub if s >= 11) / len(sub)) if sub else 0.5
    st_tai, st_xiu = streaks_tai_xiu(hist)
    feats["streak_tai"] = st_tai / 10.0
    feats["streak_xiu"] = st_xiu / 10.0
    row = markov_row_probs(last)
    feats["markov_tai_row"] = sum(p for s, p in row.items() if s >= 11)
    feats["near_modal"] = 1.0 - abs(last - 10.5) / 7.5

    sub20 = hist[-20:] if n >= 20 else hist
    feats["entropy_20"] = calculate_entropy(sub20)
    feats["autocorr_lag1"] = autocorrelation(hist, lag=1)
    feats["autocorr_lag2"] = autocorrelation(hist, lag=2)
    sub10 = hist[-10:] if n >= 10 else hist
    feats["volatility_10"] = volatility(sub10)

    return feats

# ---------------- LOGISTIC HELPERS -----------------
def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def model_predict_proba(m, feats):
    z = m.get("bias", 0.0)
    for k, v in feats.items():
        z += m["weights"].get(k, 0.0) * v
    return sigmoid(z)

# ---------------- ENSEMBLE TRAINING -----------------
def update_one_model(m, feats, label):
    p = model_predict_proba(m, feats)
    error = p - float(label)
    feat_list = list(feats.items())
    # deterministic dropout based on model id
    drop_count = int(len(feat_list) * DROPOUT_RATE)
    drop_indices = [(m["id"] + i) % len(feat_list) for i in range(drop_count)] if len(feat_list) > 0 else []
    active_feats = {k: v for i, (k, v) in enumerate(feat_list) if i not in drop_indices}

    # Update bias
    m["bias"] = m.get("bias", 0.0) - m.get("lr", 0.05) * error
    # Update weights for active feats
    for k, x in active_feats.items():
        w = m["weights"].get(k, 0.0)
        grad = error * x + L2_REG * w
        m["weights"][k] = w - m.get("lr", 0.05) * grad

def update_ensemble(feats, label):
    for m in ensemble:
        update_one_model(m, feats, label)

def update_model_scores(label):
    # Update score_ema using current state (uses last-known features)
    feats_now = features_for_state(list(history))
    for m in ensemble:
        p = model_predict_proba(m, feats_now) if history else 0.5
        pred_label = 1 if p >= 0.5 else 0
        correct = 1.0 if pred_label == label else 0.0
        prev = m.get("score_ema", 0.5)
        m["score_ema"] = prev * (1.0 - EMA_ALPHA) + correct * EMA_ALPHA

# ---------------- META-LEARNER -----------------
def update_meta(feats, label, emean):
    # Simple GD for meta weights
    p_meta = sigmoid(meta["bias"] + sum(meta["weights"].get(k, 0.0) * v for k, v in feats.items()))
    error = p_meta - label
    lr_meta = 0.0005  # small lr for meta
    meta["bias"] -= lr_meta * error
    for k, x in feats.items():
        w = meta["weights"].get(k, 0.0)
        meta["weights"][k] = w - lr_meta * (error * x + L2_REG * w)

def ensemble_proba_weighted(feats):
    ssum = 0.0
    wsum = 0.0
    for m in ensemble:
        p = model_predict_proba(m, feats)
        w = max(0.0001, m.get("score_ema", 0.5))
        ssum += p * w
        wsum += w
    # Apply meta weights to adjust
    z_meta = meta["bias"]
    for k, v in feats.items():
        z_meta += meta["weights"].get(k, 0.0) * v
    adjusted = sigmoid(z_meta) * (ssum / wsum if wsum > 0 else 0.5)
    return adjusted

# ---------------- TOP-K SUMS (DETERMINISTIC) -----------------
def top_k_sums_constrained(hist, du_doan, k=3):
    allowed = list(range(11, 18)) if du_doan == "Tài" else list(range(4, 11))
    row = markov_row_probs(hist[-1]) if hist else {s:1.0/16.0 for s in range(3,19)}
    freq = frequency_probs()
    theory = P_THEORY
    feats = features_for_state(hist)
    p_tai = ensemble_proba_weighted(feats)

    w_markov = 0.48
    w_freq = 0.26
    w_theory = 0.18
    w_ai = 0.08

    scores = {}
    for s in allowed:
        ai_bonus = p_tai if s >= 11 else (1.0 - p_tai)
        scores[s] = (w_markov * row.get(s,0.0) +
                     w_freq * freq.get(s,0.0) +
                     w_theory * theory.get(s,0.0) +
                     w_ai * ai_bonus)

    ranked = sorted(scores.items(), key=lambda x: (-x[1], abs(x[0]-10.5), x[0]))
    chosen = []
    for s, _ in ranked:
        if s not in chosen:
            chosen.append(s)
        if len(chosen) >= k:
            break
    if len(chosen) < k:
        fillers = sorted(allowed, key=lambda s: (- theory.get(s,0.0), abs(s-10.5), s))
        for s in fillers:
            if s not in chosen:
                chosen.append(s)
            if len(chosen) >= k:
                break
    chosen = chosen[:k]
    chosen.sort()
    return chosen, scores

# ---------------- PREDICTION PIPELINE (WITH CHI-SQ) -----------------
def ai_predict_pipeline(hist, top_k=3):
    feats = features_for_state(hist)
    p_tai = ensemble_proba_weighted(feats)
    du_doan = "Tài" if p_tai >= 0.5 else "Xỉu"
    do_tin_cay = f"{(p_tai if du_doan=='Tài' else 1.0-p_tai)*100:.2f}%"
    predicted_sums, scores = top_k_sums_constrained(hist, du_doan, k=top_k)

    recent10 = list(hist)[-10:] if hist else []
    num_xiu_recent = sum(1 for s in recent10 if s <= 10)

    # Chi-squared test for deviation from theory
    if len(recent10) >= 10:
        observed = Counter(recent10)
        expected = {s: len(recent10) * P_THEORY.get(s, 0) for s in range(3, 19)}
        chi2 = sum(((observed.get(s, 0) - expected[s]) ** 2 / expected[s]) for s in expected if expected[s] > 0)
        deviation = "high" if chi2 > 25 else "low"  # threshold heuristic
    else:
        deviation = "low"

    if len(recent10) >= 10 and num_xiu_recent >= 8:
        note = f"[AI Bẻ Cầu Nâng Cao] Xỉu {num_xiu_recent}/10 gần nhất + chi2 {deviation} → ưu tiên Tài."
    elif len(recent10) >= 10 and num_xiu_recent <= 2:
        note = f"[AI Bẻ Cầu Nâng Cao] Tài chiếm nhiều + chi2 {deviation} → ưu tiên Xỉu."
    else:
        note = f"[AI Phối Hợp Nâng Cao] Markov + Frequency + Theory + Ensemble({len(ensemble)}) + Meta-Weights + Feats(entropy, autocorr, volatility). Chi2 deviation: {deviation}."

    # Aggregate top feature effect (interpretability)
    feat_vals = feats
    agg_feat_scores = {}
    for m in ensemble:
        for k, w in m.get("weights", {}).items():
            agg_feat_scores[k] = agg_feat_scores.get(k, 0.0) + abs(w) * abs(feat_vals.get(k, 0.0))
    top_feats = sorted(agg_feat_scores.items(), key=lambda x: -x[1])[:5]
    top_feat_names = ", ".join([k for k, _ in top_feats]) if top_feats else "bias"

    ghi_chu = f"{note} | top_features: {top_feat_names} | p_tai={p_tai:.4f}"
    return {
        "du_doan": du_doan,
        "xac_suat_tai": round(p_tai, 4),
        "doan_vi": ", ".join(map(str, predicted_sums)),
        "do_tin_cay": do_tin_cay,
        "ghi_chu": ghi_chu,
        "scores": {str(k): round(v, 8) for k, v in scores.items()}
    }

# ---------------- HELPERS: CORS & RATE LIMIT -----------------
def cors_response(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Admin-Token'
    return resp

def rate_limit_ok(ip):
    now = time()
    ts = rate_limits[ip]
    # keep only recent
    rate_limits[ip] = [t for t in ts if now - t <= RATE_LIMIT_WINDOW]
    if len(rate_limits[ip]) >= RATE_LIMIT_MAX:
        return False
    rate_limits[ip].append(now)
    return True

# ---------------- ENDPOINTS -----------------
@app.route('/api/sicbo/sunwin', methods=['GET', 'OPTIONS'])
def get_sicbo():
    ip = request.remote_addr or "anon"
    if not rate_limit_ok(ip):
        return cors_response(jsonify({"error": "rate limit exceeded"})), 429

    # fetch remote data (try/except to avoid crash)
    try:
        with urllib.request.urlopen(API_URL, timeout=8) as resp:
            orig_data = json.loads(resp.read().decode())
    except Exception as e:
        # if remote fetch fails, still return a prediction from local history
        orig_data = {}

    global last_phien

    current_phien = orig_data.get('Phien') if orig_data else None
    current_total = orig_data.get('Tong') if orig_data else None

    # If there's a new session/result, use it to update models
    if current_phien is not None and current_total is not None and current_phien != last_phien:
        try:
            if len(history) >= 1:
                label = 1 if int(current_total) >= 11 else 0
                feats_prev = features_for_state(list(history))
                update_ensemble(feats_prev, label)
                update_model_scores(label)
                # compute ensemble mean prediction for meta update
                emean = sum(model_predict_proba(m, feats_prev) for m in ensemble) / len(ensemble) if ensemble else 0.5
                update_meta(feats_prev, label, emean)
                update_markov(history[-1], int(current_total))
            history.append(int(current_total))
            last_phien = current_phien

            # scoring historical accuracy vs previous prediction
            if len(history) >= 2:
                prev_hist = list(history)[:-1]
                prev_pred = ai_predict_pipeline(prev_hist, top_k=TOP_K_DEFAULT)
                pred_label = 1 if prev_pred['du_doan'] == "Tài" else 0
                true_label = 1 if int(current_total) >= 11 else 0
                metrics['total_predictions'] += 1
                if pred_label == true_label:
                    metrics['correct_predictions'] += 1
                    metrics['rolling'].append(1)
                else:
                    metrics['rolling'].append(0)
            # periodic save
            save_state()
        except Exception as e:
            print("Update on new session error:", e)

    # get a prediction from current history
    pred = ai_predict_pipeline(list(history), top_k=TOP_K_DEFAULT)

    resp = {
        "ai_version": "VIP-ULTRA-v3-no-numpy",
        "id": "tele@idol_vannhat",
        "session": orig_data.get('Phien', '#Unknown') if orig_data else '#Unknown',
        "dice": f"{orig_data.get('Xuc_xac_1', 0)} - {orig_data.get('Xuc_xac_2', 0)} - {orig_data.get('Xuc_xac_3', 0)}" if orig_data else "0 - 0 - 0",
        "total": orig_data.get('Tong', history[-1] if history else 0) if orig_data else (history[-1] if history else 0),
        "result": orig_data.get('Ket_qua', 'Unknown') if orig_data else 'Unknown',
        "next_session": orig_data.get('phien_hien_tai', 0) if orig_data else 0,
        "du_doan": pred.get('du_doan'),
        "doan_vi": pred.get('doan_vi'),
        "do_tin_cay": pred.get('do_tin_cay'),
        "xac_suat_tai": pred.get('xac_suat_tai'),
        "ghi_chu": pred.get('ghi_chu'),
        "scores": pred.get('scores'),
        "metrics": {
            "total_predictions": metrics["total_predictions"],
            "correct_predictions": metrics["correct_predictions"],
            "rolling_mean": (sum(metrics["rolling"]) / len(metrics["rolling"])) if metrics["rolling"] else None
        }
    }
    return cors_response(jsonify(resp))

# Simple health + admin endpoints for state & debug
@app.route('/api/sicbo/state', methods=['GET'])
def get_state():
    token = request.headers.get("X-Admin-Token", "")
    if token != ADMIN_TOKEN:
        return cors_response(jsonify({"error": "forbidden"})), 403
    # return lightweight state summary
    return cors_response(jsonify({
        "history_len": len(history),
        "last_phien": last_phien,
        "ensemble_size": len(ensemble),
        "meta_bias": meta["bias"],
        "metrics": {
            "total_predictions": metrics["total_predictions"],
            "correct_predictions": metrics["correct_predictions"]
        }
    }))

@app.route('/api/sicbo/reset', methods=['POST'])
def reset_state():
    token = request.headers.get("X-Admin-Token", "")
    if token != ADMIN_TOKEN:
        return cors_response(jsonify({"error": "forbidden"})), 403
    history.clear()
    trans.clear()
    init_ensemble()
    meta["weights"].clear()
    meta["bias"] = 0.0
    metrics["total_predictions"] = 0
    metrics["correct_predictions"] = 0
    metrics["rolling"].clear()
    save_state()
    return cors_response(jsonify({"ok": True}))

# ---------------- START -----------------
if __name__ == '__main__':
    init_ensemble()
    load_state()
    # Seed some reasonable history if empty (optional)
    if len(history) == 0:
        seed = [10, 11, 9, 12, 8, 13, 10, 11, 9, 12, 11, 10, 13, 9]
        for s in seed:
            history.append(int(s))
    # ensure lr present
    for m in ensemble:
        if "lr" not in m:
            m["lr"] = 0.05
    # run flask
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
