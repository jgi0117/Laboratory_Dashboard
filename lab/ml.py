# -*- coding: utf-8 -*-
"""정방향 예측(조성·공정 → 성능), 역방향 설계(규격 → 조성 후보), 민감도, 유사 실험.

- numpy·scikit-learn이 있을 때만 켜진다. 없으면 Engine.available == False.
- 입력 행렬의 열 순서: [숫자 입력 변수 | 범주형 입력 변수 one-hot | 프로젝트(제품군) one-hot]
  · 범주형: 값이 대부분 문자열인 입력 변수(예: 안료 종류). 2~20개 수준이면 one-hot으로 넣는다.
  · 제품군: 프로젝트마다 기본 성능이 다른 것(변수로 적히지 않은 안료·구조 차이 등)을 모델이 흡수한다.
    학습에 없는 프로젝트는 효과 0(전체 평균 기준)으로 예측하고 경고한다.
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
MAX_LEVELS = 20
MODEL_NAMES = {"rf": "RandomForest", "gbr": "GradientBoosting", "ridge": "Ridge(선형)"}
PROJECT_KEY = "__project__"
CACHE_VERSION = 2
CACHE_KEYS = ("models", "features", "cats", "project_ids", "bounds", "medians", "decimals", "X", "X_ids", "X_proj", "ysd",
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


def _present(v):
    return v is not None and str(v).strip() != ""


class NotTrained(RuntimeError):
    pass


class Engine:
    def __init__(self, store, cache_path=None):
        self.store = store
        self.cache_path = cache_path
        self.available = AVAILABLE
        self.models = {}          # target id -> dict
        self.features = []        # 숫자 입력 변수 id
        self.cats = []            # [(변수 id, [수준...]), ...]  범주형 입력
        self.project_ids = []     # one-hot 열 순서
        self.bounds = {}          # var id -> (min, max)
        self.medians = {}
        self.decimals = {}
        self.X = None             # 학습 입력 행렬 (NaN은 중앙값)
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
            if obj.get("cache_version") != CACHE_VERSION or any(k not in obj for k in CACHE_KEYS):
                return                      # 옛 형식의 캐시는 무시한다 (다시 학습하면 새로 만들어진다)
            for k in CACHE_KEYS:
                setattr(self, k, obj[k])
        except Exception:
            pass

    def _save_cache(self):
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump({"cache_version": CACHE_VERSION, **{k: getattr(self, k) for k in CACHE_KEYS}}, f)
        except OSError:
            pass

    def signature(self):
        exps = [e for e in self.store.experiments() if not e.get("excluded")]
        h = hashlib.md5(json.dumps([(e["id"], e.get("project"), e.get("values")) for e in exps], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return f"{len(exps)}:{h[:12]}"

    # ---------- 열 구성 ----------
    @property
    def n_num(self):
        return len(self.features)

    @property
    def n_cat(self):
        return sum(len(lv) for _, lv in self.cats)

    def _cat_offset(self, vid):
        k = self.n_num
        for v, levels in self.cats:
            if v == vid:
                return k
            k += len(levels)
        raise KeyError(vid)

    def _frame(self):
        vars_ = self.store.variables()
        in_ids = [v["id"] for v in dataset.inputs(vars_)]
        targets = dataset.targets(vars_)
        exps = [e for e in self.store.experiments() if not e.get("excluded")]
        rows = [e for e in exps if any(_present((e.get("values") or {}).get(i)) for i in in_ids)]
        feats, cats, dropped = [], [], []
        for vid in in_ids:
            present = [(e.get("values") or {}).get(vid) for e in rows if _present((e.get("values") or {}).get(vid))]
            if len(present) < max(MIN_ROWS, FEATURE_MIN_FRAC * len(rows)):
                dropped.append(vid)
                continue
            nums = [x for x in (to_number(p) for p in present) if x is not None]
            if len(nums) >= 0.8 * len(present):                 # 숫자 변수
                if nums and (max(nums) - min(nums)) > 0:
                    feats.append(vid)
                else:
                    dropped.append(vid)
            else:                                                # 범주형 변수
                levels = sorted({str(p).strip() for p in present if to_number(p) is None})
                if 2 <= len(levels) <= MAX_LEVELS:
                    cats.append((vid, levels))
                else:
                    dropped.append(vid)
        return rows, feats, cats, dropped, targets

    def _onehot(self, project):
        return [1.0 if project == pid else 0.0 for pid in self.project_ids]

    def _cat_vec(self, values):
        """범주형 입력 → one-hot 목록, 모르는 수준(학습에 없던 값)의 변수 id 목록."""
        out, unknown = [], []
        for vid, levels in self.cats:
            v = (values or {}).get(vid)
            s = str(v).strip() if _present(v) else ""
            if s and s not in levels:
                unknown.append(vid)
            out += [1.0 if s == l else 0.0 for l in levels]
        return out, unknown

    def _vector(self, values, project=None):
        """입력 dict → 벡터(숫자는 없으면 중앙값) + 범위 밖·누락·모르는 수준 목록."""
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
        cat, unknown = self._cat_vec(values)
        return np.array(x + cat + self._onehot(project), dtype=float), out_of_range, missing, unknown

    # ---------- 학습 ----------
    def train(self):
        if not self.available:
            raise RuntimeError("numpy·scikit-learn이 없어요.  pip install numpy scikit-learn")
        t0 = time.time()
        rows, feats, cats, dropped, targets = self._frame()
        vmap = {v["id"]: v for v in self.store.variables()}
        self.notes = []
        if len(rows) < MIN_ROWS or not (feats or cats):
            raise RuntimeError(f"학습할 실험이 부족해요 (입력값이 있는 실험 {len(rows)}건, 필요 {MIN_ROWS}건 이상)")
        if dropped:
            self.notes.append("값이 거의 없거나 늘 같은 값이라 모델에서 뺀 입력: " + ", ".join(vmap[i]["name"] for i in dropped if i in vmap))
        if cats:
            self.notes.append("범주형(문자열)으로 넣은 입력: " + ", ".join(f"{vmap[v]['name']} ({len(lv)}가지)" for v, lv in cats if v in vmap))
        self.features, self.cats = feats, cats
        self.project_ids = sorted({e.get("project") for e in rows})
        self.X_ids = [e["id"] for e in rows]
        self.X_proj = [e.get("project") for e in rows]
        nf = len(feats)
        Xv = np.array([[to_number((e.get("values") or {}).get(f)) if to_number((e.get("values") or {}).get(f)) is not None else np.nan
                        for f in feats] for e in rows], dtype=float).reshape(len(rows), nf)
        # 탐색·경고 범위는 '점검'에서 이상치로 걸린 값(오타 의심)을 뺀 범위로 잡는다
        flagged = {(c["exp"], c["var"]) for c in dataset.checks(self.store) if c["type"] == "outlier"}
        self.bounds, self.medians, self.decimals = {}, {}, {}
        for j, f in enumerate(feats):
            col = [x for e, x in zip(rows, Xv[:, j]) if not np.isnan(x) and (e["id"], f) not in flagged]
            if len(col) < 2:
                col = [x for x in Xv[:, j] if not np.isnan(x)]
            self.bounds[f] = (float(min(col)), float(max(col)))
            self.medians[f] = float(np.nanmedian(Xv[:, j]))
            self.decimals[f] = _decimals([x for x in Xv[:, j] if not np.isnan(x)])
        C = np.array([self._cat_vec(e.get("values"))[0] for e in rows], dtype=float).reshape(len(rows), self.n_cat)
        H = np.array([self._onehot(e.get("project")) for e in rows], dtype=float).reshape(len(rows), len(self.project_ids))
        X = np.hstack([Xv, C, H])
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
                # 채택 모델과 별도로 RF를 하나 더 두어 '모델이 자신 없는 정도'(트리 분산)를 어디서나 잴 수 있게 한다
                unc = est if key == "rf" else _candidates()["rf"].fit(Xi, yi)
                imp = permutation_importance(est, Xi, yi, n_repeats=5, random_state=0)
            vals = [float(max(v, 0.0)) for v in imp.importances_mean]
            imps = {f: vals[j] for j, f in enumerate(feats)}
            k = nf
            for vid, levels in cats:
                imps[vid] = float(sum(vals[k:k + len(levels)]))
                k += len(levels)
            if self.project_ids:
                imps[PROJECT_KEY] = float(sum(vals[k:]))
            tot = sum(imps.values()) or 1.0
            yspan = float(yi.max() - yi.min()) or 1.0
            clip_lo, clip_hi = float(yi.min()) - 0.15 * yspan, float(yi.max()) + 0.15 * yspan
            if str(vmap.get(tid, {}).get("unit", "")).strip() == "%" and yi.min() >= 0 and yi.max() <= 100:
                clip_lo, clip_hi = max(clip_lo, 0.0), min(clip_hi, 100.0)     # 퍼센트는 0~100 밖으로 나가지 않게
            models[tid] = {"ok": True, "n": n, "model": est, "unc": unc, "kind": key, "name": MODEL_NAMES[key], "clip": (clip_lo, clip_hi),
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
               "n_rows": len(self.X_ids), "projects": list(self.project_ids), "features": [], "categoricals": [],
               "targets": [], "notes": self.notes, "project_key": PROJECT_KEY}
        if not self.available:
            out["install"] = "pip install numpy scikit-learn"
        for f in self.features:
            v = vmap.get(f)
            if v:
                lo, hi = self.bounds[f]
                out["features"].append({"id": f, "name": v["name"], "unit": v.get("unit", ""), "kind": v["kind"],
                                        "min": lo, "max": hi, "median": self.medians[f], "decimals": self.decimals.get(f, 2)})
        for vid, levels in self.cats:
            v = vmap.get(vid)
            if v:
                out["categoricals"].append({"id": vid, "name": v["name"], "unit": v.get("unit", ""), "kind": v["kind"], "levels": list(levels)})
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
        if m.get("clip"):
            pred = np.clip(pred, m["clip"][0], m["clip"][1])
        # 구간 = 잡음(교차검증 잔차) + 모델 불확실성(트리 분산)
        sd = np.sqrt(m["resid_sd"] ** 2 + self._epistemic(Xm, tid) ** 2)
        return pred, sd

    def _epistemic(self, Xm, tid):
        """모델이 그 입력에서 얼마나 자신 없는지 (RF 트리별 예측의 표준편차). 데이터가 쌓이면 줄어든다."""
        m = self.models[tid]
        unc = m.get("unc") or (m["model"] if m["kind"] == "rf" else None)
        if unc is None:
            return np.zeros(len(Xm))
        rf = unc.steps[-1][1]
        Xt = unc[:-1].transform(Xm)
        trees = np.array([t.predict(Xt) for t in rf.estimators_])
        return trees.std(axis=0)

    def _standardize(self, Xm):
        nf = self.n_num
        out = Xm.copy()
        if nf:
            lo = np.array([self.bounds[f][0] for f in self.features])
            hi = np.array([self.bounds[f][1] for f in self.features])
            span = np.where(hi - lo > 0, hi - lo, 1.0)
            out[:, :nf] = (Xm[:, :nf] - lo) / span
        return out

    def _values_of(self, row):
        """행 벡터 → {변수 id: 값} (숫자는 float, 범주형은 수준 문자열)."""
        vals = {f: round(float(row[j]), self.decimals.get(f, 2)) for j, f in enumerate(self.features)}
        k = self.n_num
        for vid, levels in self.cats:
            seg = row[k:k + len(levels)]
            if len(seg) and seg.max() > 0:
                vals[vid] = levels[int(seg.argmax())]
            k += len(levels)
        return vals

    # ---------- 정방향 ----------
    def predict(self, values, specs=None, project=None):
        self._need()
        x, oor, missing, unknown = self._vector(values, project)
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
        return {"targets": out, "out_of_range": oor, "missing": missing, "unknown_levels": unknown,
                "project_warning": self._project_warning(project), "nearest": self.similar(values, project=project, k=3)}

    # ---------- 유사 실험 ----------
    def similar(self, values, project=None, k=5):
        if not self.models or self.X is None or not len(self.X_ids):
            return []
        x, _, missing, _ = self._vector(values, project)
        nf = self.n_num
        use = [f not in missing for f in self.features]
        if not any(use):                 # 입력이 하나도 없으면 중앙값 기준으로 가까운 실험
            use = [True] * nf
        for vid, levels in self.cats:
            use += [_present((values or {}).get(vid))] * len(levels)
        use += [bool(project)] * len(self.project_ids)
        use = np.array(use, dtype=bool)
        if not use.any():
            return []
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
            raise NotTrained("그 입력은 모델에 들어가지 않았거나 숫자 변수가 아니에요")
        x, _, _, _ = self._vector(base, project)
        j = self.features.index(var)
        lo, hi = self.bounds[var]
        grid = np.linspace(lo, hi, steps)
        Xm = np.tile(x, (steps, 1))
        Xm[:, j] = grid
        p, sd = self._predict_matrix(Xm, target)
        return {"x": grid.tolist(), "pred": p.tolist(), "lo": (p - 1.96 * sd).tolist(), "hi": (p + 1.96 * sd).tolist()}

    # ---------- 후보 생성 (역방향·파레토 공용) ----------
    def _sample_inputs(self, fixed, project, n_samples, rng):
        """데이터 범위 안의 Latin Hypercube 샘플 + 실제 실험점. 반환: 전체 입력 행렬 X, (fixed_num, fixed_cat)"""
        fixed = fixed or {}
        fixed_num = {k: to_number(v) for k, v in fixed.items() if k in self.features and to_number(v) is not None}
        cat_levels = dict(self.cats)
        fixed_cat = {k: str(v).strip() for k, v in fixed.items() if k in cat_levels and _present(v) and str(v).strip() in cat_levels[k]}
        nf, nc = self.n_num, self.n_cat
        lo = np.array([self.bounds[f][0] for f in self.features])
        hi = np.array([self.bounds[f][1] for f in self.features])

        def random_cats(m):
            blocks = []
            for vid, levels in self.cats:
                B = np.zeros((m, len(levels)))
                if vid in fixed_cat:
                    B[:, levels.index(fixed_cat[vid])] = 1.0
                else:
                    B[np.arange(m), rng.integers(0, len(levels), m)] = 1.0
                blocks.append(B)
            return np.hstack(blocks) if blocks else np.zeros((m, 0))

        U = np.empty((n_samples, nf))
        for j in range(nf):
            U[:, j] = (rng.permutation(n_samples) + rng.random(n_samples)) / n_samples
        Xv = np.vstack([lo + U * (hi - lo), self.X[:, :nf]])
        Xc = np.vstack([random_cats(n_samples), self.X[:, nf:nf + nc].copy()])
        for vid, lv in fixed_cat.items():
            k = self._cat_offset(vid) - nf
            levels = cat_levels[vid]
            Xc[:, k:k + len(levels)] = 0.0
            Xc[:, k + levels.index(lv)] = 1.0
        for j, f in enumerate(self.features):
            Xv[:, j] = np.round(Xv[:, j], self.decimals.get(f, 2))
            if f in fixed_num:
                Xv[:, j] = fixed_num[f]
        onehot = np.array(self._onehot(project), dtype=float)
        X = np.hstack([Xv, Xc, np.tile(onehot, (len(Xv), 1))])
        return X, fixed_num, fixed_cat, random_cats

    def pareto(self, objectives, specs=None, fixed=None, project=None, n_samples=3000, seed=0, max_front=12):
        """두 개 이상의 성능 사이 트레이드오프 경계(파레토 전선).
        objectives: [{"var": 성능 id, "dir": "max"|"min"}], specs: 지켜야 할 규격(선택), fixed: 고정 입력"""
        self._need()
        objs = [(o["var"], -1.0 if str(o.get("dir", "max")).lower() == "max" else 1.0)
                for o in objectives or [] if o.get("var") in self.models and self.models[o["var"]]["ok"]]
        if len(objs) < 2:
            raise NotTrained("비교할 성능을 두 개 이상 고르세요 (학습된 모델이 있는 성능만)")
        parsed = []
        for s in specs or []:
            sp = dataset.parse_spec(s.get("spec"))
            if sp and s.get("var") in self.models and self.models[s["var"]]["ok"]:
                parsed.append((s["var"], sp))
        rng = np.random.default_rng(seed)
        X, fixed_num, fixed_cat, _ = self._sample_inputs(fixed, project, n_samples, rng)
        preds = {tid: self._predict_matrix(X, tid) for tid in {t for t, _ in objs} | {t for t, _ in parsed}}
        ok = np.ones(len(X), dtype=bool)
        for tid, sp in parsed:
            ok &= np.array([dataset.judge(sp["text"], float(v)) != "Fail" for v in preds[tid][0]])
        constrained = bool(parsed) and bool(ok.any())
        idx = np.where(ok)[0] if constrained else np.arange(len(X))
        M = np.column_stack([preds[tid][0][idx] * sgn for tid, sgn in objs])      # 모두 '작을수록 좋다'로
        # 파레토 전선: 어떤 점에도 지배당하지 않는 점
        order = np.lexsort(M.T[::-1])
        front = []
        for i in order:
            if not any(np.all(M[j] <= M[i]) and np.any(M[j] < M[i]) for j in front):
                front.append(i)
        front = [idx[i] for i in front]
        # 전선이 너무 길면 첫 목표 기준으로 고르게 솎는다
        if len(front) > max_front:
            fv = preds[objs[0][0]][0][front]
            keep = np.linspace(0, len(front) - 1, max_front).round().astype(int)
            front = [front[i] for i in np.argsort(fv)[keep]]
        vmap = {v["id"]: v for v in self.store.variables()}

        def point(i):
            return {"values": self._values_of(X[i]),
                    "objectives": [{"id": tid, "name": vmap[tid]["name"], "unit": vmap[tid].get("unit", ""),
                                    "pred": float(preds[tid][0][i]), "sd": float(preds[tid][1][i])} for tid, _ in objs],
                    "specs_ok": bool(ok[i]) if parsed else None}
        front_pts = sorted((point(i) for i in front), key=lambda p: p["objectives"][0]["pred"])
        cloud_idx = rng.choice(len(X), size=min(400, len(X)), replace=False)
        cloud = [{"x": float(preds[objs[0][0]][0][i]), "y": float(preds[objs[1][0]][0][i]), "ok": bool(ok[i]) if parsed else True} for i in cloud_idx]
        return {"front": front_pts, "cloud": cloud, "constrained": constrained, "n_feasible": int(ok.sum()) if parsed else len(X),
                "fixed": {**fixed_num, **fixed_cat}, "project_warning": self._project_warning(project),
                "message": "" if (constrained or not parsed) else "규격을 모두 만족하는 점이 없어서 규격을 무시하고 경계를 그렸어요."}

    # ---------- 탐색 실험 추천 (모델이 가장 불확실한 곳) ----------
    def explore(self, specs=None, fixed=None, project=None, n=3, n_samples=3000, seed=0):
        """다음에 해 볼 실험 추천: 규격(있으면)을 만족할 것으로 보이면서 예측 불확실성이 가장 큰, 서로 다른 조성 n개.
        이런 실험을 하면 모델이 가장 많이 배운다 (능동 학습)."""
        self._need()
        parsed = []
        for s in specs or []:
            sp = dataset.parse_spec(s.get("spec"))
            if sp and s.get("var") in self.models and self.models[s["var"]]["ok"]:
                parsed.append((s["var"], sp))
        rng = np.random.default_rng(seed)
        X, fixed_num, fixed_cat, _ = self._sample_inputs(fixed, project, n_samples, rng)
        ok_models = [tid for tid, m in self.models.items() if m["ok"]]
        preds = {tid: self._predict_matrix(X, tid) for tid in ok_models}
        unc = np.zeros(len(X))
        for tid in ok_models:
            unc += self._epistemic(X, tid) / (self.ysd.get(tid) or 1.0)     # 잡음 말고 '모델이 모르는 정도'만
        unc /= max(len(ok_models), 1)
        feas = np.ones(len(X), dtype=bool)
        for tid, sp in parsed:
            feas &= np.array([dataset.judge(sp["text"], float(v)) != "Fail" for v in preds[tid][0]])
        if not feas.any():
            feas[:] = True
        S = self._standardize(X)
        S_train = self._standardize(self.X)
        dist = np.sqrt(((S[:, None, :] - S_train[None, :, :]) ** 2).mean(axis=2)).min(axis=1)
        score = np.where(feas, unc + 0.5 * dist, -np.inf)
        order = np.argsort(-score)
        chosen = []
        for i in order:
            if not np.isfinite(score[i]):
                break
            if all(np.sqrt(((S[i] - S[k]) ** 2).mean()) > 0.15 for k in chosen):
                chosen.append(i)
            if len(chosen) >= n:
                break
        vmap = {v["id"]: v for v in self.store.variables()}
        out = []
        for i in chosen:
            values = self._values_of(X[i])
            targets = []
            for tid in ok_models:
                p, sd = float(preds[tid][0][i]), float(preds[tid][1][i])
                ep = float(self._epistemic(X[i:i + 1], tid)[0])
                item = {"id": tid, "name": vmap[tid]["name"], "unit": vmap[tid].get("unit", ""), "pred": p, "sd": sd, "epistemic": ep,
                        "lo": p - 1.96 * sd, "hi": p + 1.96 * sd, "rel": ep / (self.ysd.get(tid) or 1.0)}
                sp = next((s for t, s in parsed if t == tid), None)
                if sp:
                    item["spec"], item["result"] = sp["text"], dataset.judge(sp["text"], p)
                targets.append(item)
            targets.sort(key=lambda t: -t["rel"])
            out.append({"values": values, "targets": targets, "uncertainty": round(float(unc[i]), 3), "distance": round(float(dist[i]), 3),
                        "nearest": self.similar(values, project=project, k=1)})
        return {"candidates": out, "fixed": {**fixed_num, **fixed_cat}, "project_warning": self._project_warning(project),
                "constrained": bool(parsed)}

    # ---------- 역방향 ----------
    def inverse(self, specs, fixed=None, project=None, n=5, n_samples=2000, seed=0, prefer_known=0.3):
        """규격을 만족하는 입력 후보 찾기.
        specs: [{"var": 성능 id, "spec": "≥70"}], fixed: {입력 id: 값(숫자 또는 범주 수준)}, project: 제품군 효과에 쓸 프로젝트
        점수 = 2·Σ 정규화 위반량 + 0.2·Σ 정규화 불확실성 + prefer_known·데이터와의 거리  (작을수록 좋다)"""
        self._need()
        parsed = []
        for s in specs or []:
            sp = dataset.parse_spec(s.get("spec"))
            tid = s.get("var")
            if sp and tid in self.models and self.models[tid]["ok"]:
                parsed.append((tid, sp))
        if not parsed:
            raise NotTrained("규격이 없거나, 규격에 적은 성능의 모델이 없어요")
        rng = np.random.default_rng(seed)
        nf, nc = self.n_num, self.n_cat
        lo = np.array([self.bounds[f][0] for f in self.features])
        hi = np.array([self.bounds[f][1] for f in self.features])
        X, fixed_num, fixed_cat, random_cats = self._sample_inputs(fixed, project, n_samples, rng)
        free = np.array([f not in fixed_num for f in self.features], dtype=bool)
        S_train = self._standardize(self.X)

        def score(Xm):
            total = np.zeros(len(Xm))
            preds = {}
            for tid, sp in parsed:
                p, sd = self._predict_matrix(Xm, tid)
                ysd = self.ysd.get(tid) or 1.0
                viol = np.array([dataset.violation(sp, float(v)) for v in p]) / ysd
                total += 2.0 * viol + 0.2 * sd / ysd
                preds[tid] = (p, sd)
            S = self._standardize(Xm)
            d = np.sqrt(((S[:, None, :] - S_train[None, :, :]) ** 2).mean(axis=2)).min(axis=1)
            return total + prefer_known * d, preds, d

        sc, _, _ = score(X)
        top = np.argsort(sc)[:30]
        Cm, Cs = X[top].copy(), sc[top].copy()
        span = hi - lo
        for _ in range(30):
            P = Cm.copy()
            if nf:
                noise = rng.normal(0, 0.05, (len(Cm), nf)) * span
                noise[:, ~free] = 0
                P[:, :nf] = np.clip(P[:, :nf] + noise, lo, hi)
            if nc:                                                   # 범주는 가끔 다른 수준으로 바꿔 본다
                flip = rng.random(len(Cm)) < 0.15
                if flip.any():
                    P[flip, nf:nf + nc] = random_cats(int(flip.sum()))
            ps, _, _ = score(P)
            better = ps < Cs
            Cm[better], Cs[better] = P[better], ps[better]
        for j, f in enumerate(self.features):
            Cm[:, j] = np.round(Cm[:, j], self.decimals.get(f, 2))
        Cs, preds, dist = score(Cm)
        fails = np.zeros(len(Cm), dtype=int)                        # 규격을 다 만족하는 후보가 항상 앞에
        for tid, sp in parsed:
            fails += np.array([dataset.judge(sp["text"], float(v)) == "Fail" for v in preds[tid][0]], dtype=int)
        order = np.lexsort((Cs, fails > 0))
        SC = self._standardize(Cm)
        chosen = []
        for thr in (0.08, 0.03, 0.0):                                # 서로 다른 후보를 우선, 모자라면 가까운 것도
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
            values = self._values_of(Cm[i])
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
                    p, sd = self._predict_matrix(Cm[i:i + 1], tid)
                    others.append({"id": tid, "name": vmap[tid]["name"], "unit": vmap[tid].get("unit", ""), "pred": float(p[0]), "sd": float(sd[0])})
            out.append({"values": values, "targets": targets, "others": others, "score": round(float(Cs[i]), 4),
                        "all_pass": all_pass, "distance": round(float(dist[i]), 3), "nearest": self.similar(values, project=project, k=1)})
        feasible = any(c["all_pass"] for c in out)
        return {"candidates": out, "feasible": feasible, "fixed": {**fixed_num, **fixed_cat}, "project_warning": self._project_warning(project),
                "message": "" if feasible else "데이터 범위 안에서는 모든 규격을 동시에 만족하는 조성을 찾지 못했어요. 가장 가까운 후보를 보여 드려요. 규격을 완화하거나 새 범위의 실험이 필요해요."}
