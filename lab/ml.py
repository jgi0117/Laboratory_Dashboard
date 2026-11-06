# -*- coding: utf-8 -*-
"""정방향 예측(조성·공정 → 성능), 역방향 설계(규격 → 조성 후보), 민감도, 유사 실험.

- numpy·scikit-learn이 있을 때만 켜진다. 없으면 Engine.available == False.
- 입력 = 조성·공정 변수(숫자) + 프로젝트(제품군) one-hot. 프로젝트마다 기본 성능이 다른 것(안료·구조 차이 등,
  변수로 적히지 않은 것)을 모델이 '제품군 효과'로 흡수한다. 학습에 없는 프로젝트는 효과 0(전체 평균 기준)으로 예측하고 경고한다.
"""
import hashlib
import json
import math
import pickle
import time
import warnings

from . import dataset
from .store import to_number

try:
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    AVAILABLE = True
except ImportError:          # pragma: no cover
    np = None
    AVAILABLE = False

MIN_ROWS = 8            # 이보다 적으면 그 성능은 학습하지 않는다
FEATURE_MIN_FRAC = 0.5  # 입력 변수는 X가 있는 행의 절반 이상에 값이 있어야 쓴다
MODEL_NAMES = {"rf": "RandomForest", "gbr": "GradientBoosting", "ridge": "Ridge(선형)"}
PROJECT_KEY = "__project__"
CACHE_KEYS = ("models", "features", "project_ids", "bounds", "medians", "decimals", "X", "X_ids", "X_proj", "ysd",
              "trained_at", "data_sig", "notes")


def _candidates():
    return {
        "rf": make_pipeline(SimpleImputer(strategy="median"),
                            RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=0)),
        "gbr": make_pipeline(SimpleImputer(strategy="median"),
                             GradientBoostingRegressor(n_estimators=200, learning_rate=0.06, max_depth=3, subsample=0.8, random_state=0)),
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
    }


def _decimals(values, cap=3):
    d = 0
    for v in values:
        s = repr(float(v))
        if "e" in s or "." not in s:
            continue
        d = max(d, len(s.split(".")[1].rstrip("0")))
    return min(d, cap)


class NotTrained(RuntimeError):
    pass


class Engine:
    def __init__(self, store, cache_path=None):
        self.store = store
        self.cache_path = cache_path
        self.available = AVAILABLE
        self.models = {}          # target id -> dict
        self.features = []        # 입력 변수 id (숫자)
        self.project_ids = []     # one-hot 열 순서
        self.bounds = {}          # var id -> (min, max)
        self.medians = {}
        self.decimals = {}
        self.X = None             # 학습 입력 행렬 (변수 열 + one-hot 열, NaN은 중앙값)
        self.X_ids = []           # 행별 실험 id
        self.X_proj = []          # 행별 프로젝트 id
        self.ysd = {}
        self.trained_at = None
        self.data_sig = None
        self.notes = []
        if cache_path:
            self._load_cache()

    # ---------- 캐시 ----------
    def _load_cache(self):
        try:
            with open(self.cache_path, "rb") as f:
                obj = pickle.load(f)
            for k in CACHE_KEYS:
                setattr(self, k, obj[k])
        except Exception:
            pass

    def _save_cache(self):
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump({k: getattr(self, k) for k in CACHE_KEYS}, f)
        except OSError:
            pass

    def signature(self):
        exps = [e for e in self.store.experiments() if not e.get("excluded")]
        h = hashlib.md5(json.dumps([(e["id"], e.get("project"), e.get("values")) for e in exps], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return f"{len(exps)}:{h[:12]}"

    # ---------- 데이터 준비 ----------
    def _frame(self):
        vars_ = self.store.variables()
        in_ids = [v["id"] for v in dataset.inputs(vars_)]
        targets = dataset.targets(vars_)
        exps = [e for e in self.store.experiments() if not e.get("excluded")]
        rows = [e for e in exps if any(to_number((e.get("values") or {}).get(i)) is not None for i in in_ids)]
        feats = []
        for vid in in_ids:
            got = [x for x in (to_number((e.get("values") or {}).get(vid)) for e in rows) if x is not None]
            if len(got) >= max(MIN_ROWS, FEATURE_MIN_FRAC * len(rows)) and (max(got) - min(got)) > 0:
                feats.append(vid)
        return rows, feats, in_ids, targets

    def _onehot(self, project):
        return [1.0 if project == pid else 0.0 for pid in self.project_ids]

    # ---------- 학습 ----------
    def train(self):
        if not self.available:
            raise RuntimeError("numpy·scikit-learn이 없어요.  pip install numpy scikit-learn")
        t0 = time.time()
        rows, feats, in_ids, targets = self._frame()
        vmap = {v["id"]: v for v in self.store.variables()}
        self.notes = []
        if len(rows) < MIN_ROWS or not feats:
            raise RuntimeError(f"학습할 실험이 부족해요 (입력값이 있는 실험 {len(rows)}건, 필요 {MIN_ROWS}건 이상)")
        dropped = [vmap[i]["name"] for i in in_ids if i not in feats]
        if dropped:
            self.notes.append("값이 거의 없거나 늘 같은 값이라 모델에서 뺀 입력: " + ", ".join(dropped))
        self.features = feats
        self.project_ids = sorted({e.get("project") for e in rows})
        self.X_ids = [e["id"] for e in rows]
        self.X_proj = [e.get("project") for e in rows]
        nf = len(feats)
        Xv = np.array([[to_number((e.get("values") or {}).get(f)) if to_number((e.get("values") or {}).get(f)) is not None else np.nan
                        for f in feats] for e in rows], dtype=float)
        # 탐색·경고 범위는 '점검'에서 이상치로 걸린 값(오타 의심)을 뺀 범위로 잡는다
        flagged = {(c["exp"], c["var"]) for c in dataset.checks(self.store) if c["type"] == "outlier"}
        self.bounds = {}
        for j, f in enumerate(feats):
            col = [x for e, x in zip(rows, Xv[:, j]) if not np.isnan(x) and (e["id"], f) not in flagged]
            if len(col) < 2:
                col = [x for x in Xv[:, j] if not np.isnan(x)]
            self.bounds[f] = (float(min(col)), float(max(col)))
        self.medians = {f: float(np.nanmedian(Xv[:, j])) for j, f in enumerate(feats)}
        self.decimals = {f: _decimals([x for x in Xv[:, j] if not np.isnan(x)]) for j, f in enumerate(feats)}
        H = np.array([self._onehot(e.get("project")) for e in rows], dtype=float)
        X = np.hstack([Xv, H])
        Xf = X.copy()
        for j, f in enumerate(feats):
            Xf[np.isnan(Xf[:, j]), j] = self.medians[f]
        self.X = Xf
        models = {}
        for tv in targets:
            tid = tv["id"]
            y = np.array([to_number((e.get("values") or {}).get(tid)) if to_number((e.get("values") or {}).get(tid)) is not None else np.nan
                          for e in rows], dtype=float)
            mask = ~np.isnan(y)
            n = int(mask.sum())
            if n < MIN_ROWS:
                models[tid] = {"ok": False, "n": n, "reason": f"값이 있는 실험이 {n}건이라 학습하지 않았어요 (최소 {MIN_ROWS}건)"}
                continue
            Xi, yi = X[mask], y[mask]
            if float(np.std(yi)) == 0.0:
                models[tid] = {"ok": False, "n": n, "reason": "값이 모두 같아서 학습할 것이 없어요"}
                continue
            kf = KFold(n_splits=min(5, n), shuffle=True, random_state=0)
            best, scores = None, {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for key, est in _candidates().items():
                    pred = cross_val_predict(est, Xi, yi, cv=kf)
                    resid = yi - pred
                    ss_res, ss_tot = float((resid ** 2).sum()), float(((yi - yi.mean()) ** 2).sum())
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                    mae, rmse = float(np.abs(resid).mean()), float(math.sqrt((resid ** 2).mean()))
                    scores[key] = {"r2": round(r2, 3), "mae": round(mae, 4), "rmse": round(rmse, 4)}
                    if best is None or r2 > best[1]:
                        best = (key, r2, mae, rmse, float(resid.std()))
                key, r2, mae, rmse, rsd = best
                est = _candidates()[key].fit(Xi, yi)
                imp = permutation_importance(est, Xi, yi, n_repeats=5, random_state=0)
            vals = [float(max(v, 0.0)) for v in imp.importances_mean]
            imps = {f: vals[j] for j, f in enumerate(feats)}
            if self.project_ids:
                imps[PROJECT_KEY] = float(sum(vals[nf:]))
            tot = sum(imps.values()) or 1.0
            models[tid] = {"ok": True, "n": n, "model": est, "kind": key, "name": MODEL_NAMES[key],
                           "r2": round(r2, 3), "mae": round(mae, 4), "rmse": round(rmse, 4), "resid_sd": rsd,
                           "scores": scores, "importance": {f: round(v / tot, 4) for f, v in imps.items()},
                           "y_range": (float(yi.min()), float(yi.max())), "y_sd": float(yi.std())}
            self.ysd[tid] = float(yi.std())
        self.models = models
        self.trained_at = time.strftime("%Y-%m-%d %H:%M")
        self.data_sig = self.signature()
        self._save_cache()
        return {"seconds": round(time.time() - t0, 1), **self.status()}

    # ---------- 상태 ----------
    def status(self):
        vmap = {v["id"]: v for v in self.store.variables()}
        out = {"available": self.available, "trained": bool(self.models), "trained_at": self.trained_at,
               "stale": bool(self.models) and self.data_sig != self.signature(),
               "n_rows": len(self.X_ids), "projects": list(self.project_ids), "features": [], "targets": [], "notes": self.notes,
               "project_key": PROJECT_KEY}
        if not self.available:
            out["install"] = "pip install numpy scikit-learn"
        for f in self.features:
            v = vmap.get(f)
            if v:
                lo, hi = self.bounds[f]
                out["features"].append({"id": f, "name": v["name"], "unit": v.get("unit", ""), "kind": v["kind"],
                                        "min": lo, "max": hi, "median": self.medians[f], "decimals": self.decimals.get(f, 2)})
        for tid, m in self.models.items():
            v = vmap.get(tid)
            if not v:
                continue
            item = {"id": tid, "name": v["name"], "unit": v.get("unit", ""), "ok": m["ok"], "n": m["n"]}
            if m["ok"]:
                item.update(model=m["name"], r2=m["r2"], mae=m["mae"], rmse=m["rmse"], scores=m["scores"],
                            importance=m["importance"], y_range=m["y_range"])
            else:
                item["reason"] = m["reason"]
            out["targets"].append(item)
        return out

    def _need(self):
        if not self.available:
            raise NotTrained("numpy·scikit-learn이 없어요.  pip install numpy scikit-learn")
        if not self.models:
            raise NotTrained("아직 학습하지 않았어요. '학습' 버튼을 눌러 주세요")

    # ---------- 공통 ----------
    def _vector(self, values, project=None):
        """입력 dict → 벡터(없으면 중앙값) + 범위 밖·누락 목록."""
        x, out_of_range, missing = [], [], []
        for f in self.features:
            v = to_number((values or {}).get(f))
            if v is None:
                v = self.medians[f]
                missing.append(f)
            else:
                lo, hi = self.bounds[f]
                span = (hi - lo) or 1.0
                if v < lo - 0.02 * span or v > hi + 0.02 * span:
                    out_of_range.append(f)
            x.append(v)
        return np.array(x + self._onehot(project), dtype=float), out_of_range, missing

    def _project_warning(self, project):
        if project and project not in self.project_ids:
            return "이 프로젝트의 실험은 아직 학습에 없어서, 제품군 효과 없이(전체 평균 기준) 예측해요. 실험을 몇 건 넣고 다시 학습하면 정확해져요."
        if not project and self.project_ids:
            return "프로젝트를 고르지 않아 제품군 효과 없이(전체 평균 기준) 예측해요."
        return ""

    def _predict_matrix(self, Xm, tid):
        m = self.models[tid]
        est = m["model"]
        pred = est.predict(Xm)
        sd = np.full(len(Xm), m["resid_sd"])
        if m["kind"] == "rf":
            rf = est.steps[-1][1]
            Xt = est[:-1].transform(Xm)
            trees = np.array([t.predict(Xt) for t in rf.estimators_])
            sd = np.maximum(sd, trees.std(axis=0))
        return pred, sd

    def _standardize(self, Xm):
        nf = len(self.features)
        lo = np.array([self.bounds[f][0] for f in self.features])
        hi = np.array([self.bounds[f][1] for f in self.features])
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        out = Xm.copy()
        out[:, :nf] = (Xm[:, :nf] - lo) / span
        return out

    # ---------- 정방향 ----------
    def predict(self, values, specs=None, project=None):
        self._need()
        x, oor, missing = self._vector(values, project)
        Xm = x.reshape(1, -1)
        spec_map = {s["var"]: s["spec"] for s in (specs or []) if s.get("var")}
        vmap = {v["id"]: v for v in self.store.variables()}
        out = []
        for tid, m in self.models.items():
            if tid not in vmap:
                continue
            if not m["ok"]:
                out.append({"id": tid, "name": vmap[tid]["name"], "unit": vmap[tid].get("unit", ""), "ok": False, "reason": m["reason"]})
                continue
            p, sd = self._predict_matrix(Xm, tid)
            p, sd = float(p[0]), float(sd[0])
            item = {"id": tid, "name": vmap[tid]["name"], "unit": vmap[tid].get("unit", ""), "ok": True,
                    "pred": p, "lo": p - 1.96 * sd, "hi": p + 1.96 * sd, "sd": sd, "model": m["name"], "r2": m["r2"], "mae": m["mae"]}
            if tid in spec_map:
                item["spec"] = spec_map[tid]
                item["result"] = dataset.judge(spec_map[tid], p)
                item["result_lo"] = dataset.judge(spec_map[tid], item["lo"])
                item["result_hi"] = dataset.judge(spec_map[tid], item["hi"])
            out.append(item)
        return {"targets": out, "out_of_range": oor, "missing": missing, "project_warning": self._project_warning(project),
                "nearest": self.similar(values, project=project, k=3)}

    # ---------- 유사 실험 ----------
    def similar(self, values, project=None, k=5):
        if not self.models or self.X is None or not len(self.X_ids):
            return []
        x, _, missing = self._vector(values, project)
        nf = len(self.features)
        use = np.array([f not in missing for f in self.features] + [bool(project)] * len(self.project_ids))
        if not use[:nf].any():          # 입력이 하나도 없으면 중앙값 기준으로 가까운 실험
            use[:nf] = True
        S = self._standardize(self.X)[:, use]
        s = self._standardize(x.reshape(1, -1))[0, use]
        d = np.sqrt(((S - s) ** 2).mean(axis=1))
        order = np.argsort(d)[:k]
        emap = {e["id"]: e for e in self.store.experiments()}
        pmap = {p["id"]: p["name"] for p in self.store.projects()}
        out = []
        for i in order:
            e = emap.get(self.X_ids[i])
            if e:
                out.append({"id": e["id"], "label": e.get("label", ""), "date": e.get("date", ""), "project": e.get("project"),
                            "project_name": pmap.get(e.get("project"), ""), "distance": round(float(d[i]), 3), "values": e.get("values", {})})
        return out

    # ---------- 민감도 ----------
    def sensitivity(self, target, var, base=None, project=None, steps=25):
        self._need()
        if target not in self.models or not self.models[target]["ok"]:
            raise NotTrained("그 성능은 학습된 모델이 없어요")
        if var not in self.features:
            raise NotTrained("그 입력은 모델에 들어가지 않았어요")
        x, _, _ = self._vector(base, project)
        j = self.features.index(var)
        lo, hi = self.bounds[var]
        grid = np.linspace(lo, hi, steps)
        Xm = np.tile(x, (steps, 1))
        Xm[:, j] = grid
        p, sd = self._predict_matrix(Xm, target)
        return {"x": grid.tolist(), "pred": p.tolist(), "lo": (p - 1.96 * sd).tolist(), "hi": (p + 1.96 * sd).tolist()}

    # ---------- 역방향 ----------
    def inverse(self, specs, fixed=None, project=None, n=5, n_samples=2000, seed=0, prefer_known=0.3):
        """규격을 만족하는 입력 후보 찾기.
        specs: [{"var": 성능 id, "spec": "≥70"}], fixed: {입력 id: 값}, project: 제품군 효과에 쓸 프로젝트
        점수 = Σ 정규화 위반량 + 0.2·Σ 정규화 불확실성 + prefer_known·데이터와의 거리  (작을수록 좋다)"""
        self._need()
        fixed = {k: to_number(v) for k, v in (fixed or {}).items() if k in self.features and to_number(v) is not None}
        parsed = []
        for s in specs or []:
            sp = dataset.parse_spec(s.get("spec"))
            tid = s.get("var")
            if sp and tid in self.models and self.models[tid]["ok"]:
                parsed.append((tid, sp))
        if not parsed:
            raise NotTrained("규격이 없거나, 규격에 적은 성능의 모델이 없어요")
        rng = np.random.default_rng(seed)
        nf = len(self.features)
        lo = np.array([self.bounds[f][0] for f in self.features])
        hi = np.array([self.bounds[f][1] for f in self.features])
        free = np.array([f not in fixed for f in self.features])
        onehot = np.array(self._onehot(project), dtype=float)
        U = np.empty((n_samples, nf))
        for j in range(nf):
            U[:, j] = (rng.permutation(n_samples) + rng.random(n_samples)) / n_samples
        Xv = lo + U * (hi - lo)
        Xk = self.X[:, :nf].copy()          # 실제 실험점도 출발점으로 넣는다
        Xv = np.vstack([Xv, Xk])
        for j, f in enumerate(self.features):
            if f in fixed:
                Xv[:, j] = fixed[f]
        X = np.hstack([Xv, np.tile(onehot, (len(Xv), 1))])
        S_train = self._standardize(self.X)

        def score(Xc):
            total = np.zeros(len(Xc))
            preds = {}
            for tid, sp in parsed:
                p, sd = self._predict_matrix(Xc, tid)
                ysd = self.ysd.get(tid) or 1.0
                viol = np.array([dataset.violation(sp, float(v)) for v in p]) / ysd
                total += 2.0 * viol + 0.2 * sd / ysd
                preds[tid] = (p, sd)
            S = self._standardize(Xc)
            d = np.sqrt(((S[:, None, :] - S_train[None, :, :]) ** 2).mean(axis=2)).min(axis=1)
            return total + prefer_known * d, preds, d

        sc, _, _ = score(X)
        top = np.argsort(sc)[:30]
        C, Cs = X[top].copy(), sc[top].copy()
        span = hi - lo
        for _ in range(30):
            noise = np.zeros(C.shape)
            noise[:, :nf] = rng.normal(0, 0.05, (len(C), nf)) * span
            noise[:, :nf][:, ~free] = 0
            P = C + noise
            P[:, :nf] = np.clip(P[:, :nf], lo, hi)
            ps, _, _ = score(P)
            better = ps < Cs
            C[better], Cs[better] = P[better], ps[better]
        for j, f in enumerate(self.features):
            C[:, j] = np.round(C[:, j], self.decimals.get(f, 2))
        Cs, preds, dist = score(C)
        # 모든 규격을 만족하는 후보가 항상 앞에 오도록 (그 안에서는 점수 순)
        fails = np.zeros(len(C), dtype=int)
        for tid, sp in parsed:
            fails += np.array([dataset.judge(sp["text"], float(v)) == "Fail" for v in preds[tid][0]], dtype=int)
        order = np.lexsort((Cs, fails > 0))
        SC = self._standardize(C)
        chosen = []
        for thr in (0.08, 0.03, 0.0):          # 서로 다른 후보를 우선, 모자라면 가까운 것도 채운다
            for i in order:
                if i in chosen:
                    continue
                if all(np.sqrt(((SC[i] - SC[k]) ** 2).mean()) > thr for k in chosen):
                    chosen.append(i)
                if len(chosen) >= n:
                    break
            if len(chosen) >= n:
                break
        vmap = {v["id"]: v for v in self.store.variables()}
        out = []
        for i in chosen:
            values = {f: float(C[i, j]) for j, f in enumerate(self.features)}
            targets, all_pass = [], True
            for tid, sp in parsed:
                p, sd = float(preds[tid][0][i]), float(preds[tid][1][i])
                res = dataset.judge(sp["text"], p)
                all_pass = all_pass and res != "Fail"
                targets.append({"id": tid, "name": vmap[tid]["name"], "unit": vmap[tid].get("unit", ""), "spec": sp["text"],
                                "pred": p, "lo": p - 1.96 * sd, "hi": p + 1.96 * sd, "result": res,
                                "result_lo": dataset.judge(sp["text"], p - 1.96 * sd), "result_hi": dataset.judge(sp["text"], p + 1.96 * sd)})
            others = []
            for tid, m in self.models.items():
                if m["ok"] and tid in vmap and tid not in {t for t, _ in parsed}:
                    p, sd = self._predict_matrix(C[i:i + 1], tid)
                    others.append({"id": tid, "name": vmap[tid]["name"], "unit": vmap[tid].get("unit", ""), "pred": float(p[0]), "sd": float(sd[0])})
            out.append({"values": values, "targets": targets, "others": others, "score": round(float(Cs[i]), 4),
                        "all_pass": all_pass, "distance": round(float(dist[i]), 3), "nearest": self.similar(values, project=project, k=1)})
        feasible = any(c["all_pass"] for c in out)
        return {"candidates": out, "feasible": feasible, "fixed": fixed, "project_warning": self._project_warning(project),
                "message": "" if feasible else "데이터 범위 안에서는 모든 규격을 동시에 만족하는 조성을 찾지 못했어요. 가장 가까운 후보를 보여 드려요. 규격을 완화하거나 새 범위의 실험이 필요해요."}
