# -*- coding: utf-8 -*-
"""통합 표(실험 1건 = 1행), 규격 파싱·판정, 변수 통계, 내보내기."""
import csv
import io
import math
import re
from .store import KINDS, KIND_LABEL, to_number

KIND_ORDER = {k: i for i, k in enumerate(KINDS)}


# ---------- 규격 ----------
def parse_spec(text):
    """'≥70' '<=1.5' '70~80' '100±5' '=70' '70' → {"min","max","target","tol","text"} 또는 None."""
    t = str(text or "").strip().replace(" ", "")
    if not t:
        return None
    t = t.replace("≧", "≥").replace("≦", "≤").replace(">=", "≥").replace("<=", "≤").replace("=<", "≤").replace("=>", "≥")
    t = t.replace("이상", "≥").replace("이하", "≤").replace("±", "+-").replace("－", "-").replace("−", "-")
    num = r"-?\d+(?:\.\d+)?"
    m = re.fullmatch(rf"≥({num})", t) or re.fullmatch(rf"({num})≥", t)
    if m:
        return {"min": float(m.group(1)), "max": None, "target": None, "tol": None, "text": f"≥{m.group(1)}"}
    m = re.fullmatch(rf"≤({num})", t) or re.fullmatch(rf"({num})≤", t)
    if m:
        return {"min": None, "max": float(m.group(1)), "target": None, "tol": None, "text": f"≤{m.group(1)}"}
    m = re.fullmatch(rf"({num})\+-({num})", t)
    if m:
        c, d = float(m.group(1)), abs(float(m.group(2)))
        return {"min": c - d, "max": c + d, "target": c, "tol": d, "text": f"{m.group(1)}±{m.group(2)}"}
    m = re.fullmatch(rf"({num})[~\-–]({num})", t)
    if m:
        a, b = sorted((float(m.group(1)), float(m.group(2))))
        return {"min": a, "max": b, "target": (a + b) / 2, "tol": (b - a) / 2, "text": f"{m.group(1)}~{m.group(2)}"}
    m = re.fullmatch(rf"=?({num})", t)
    if m:
        c = float(m.group(1))
        return {"min": None, "max": None, "target": c, "tol": None, "text": f"={m.group(1)}"}
    return None


def judge(spec_text, value):
    """'Pass' | 'Fail' | '' (판정 불가)."""
    sp = parse_spec(spec_text)
    v = to_number(value)
    if not sp or v is None:
        return ""
    if sp["min"] is None and sp["max"] is None:
        return ""          # 목표값만 있으면 판정하지 않는다
    if sp["min"] is not None and v < sp["min"] - 1e-9:
        return "Fail"
    if sp["max"] is not None and v > sp["max"] + 1e-9:
        return "Fail"
    return "Pass"


def violation(spec, value):
    """규격을 얼마나 벗어났는지 (같은 단위). 안이면 0."""
    if spec["min"] is not None and value < spec["min"]:
        return spec["min"] - value
    if spec["max"] is not None and value > spec["max"]:
        return value - spec["max"]
    if spec["min"] is None and spec["max"] is None and spec["target"] is not None:
        return abs(value - spec["target"])
    return 0.0


# ---------- 변수 정렬·라벨 ----------
def label(v):
    return v["name"] + (f" ({v['unit']})" if v.get("unit") else "")


def sorted_variables(variables):
    return sorted(variables, key=lambda v: (KIND_ORDER.get(v.get("kind"), 9), v["id"]))


def inputs(variables):
    return [v for v in sorted_variables(variables) if v.get("kind") != "property"]


def targets(variables):
    return [v for v in sorted_variables(variables) if v.get("kind") == "property"]


# ---------- 통합 표 ----------
def table(store, project_ids=None, include_excluded=True):
    pmap = {p["id"]: p for p in store.projects()}
    cols = sorted_variables(store.variables())
    rows = []
    for e in store.experiments():
        if project_ids and e.get("project") not in project_ids:
            continue
        if not include_excluded and e.get("excluded"):
            continue
        p = pmap.get(e.get("project"), {})
        rows.append({"id": e["id"], "project": e.get("project"), "project_name": p.get("name", ""),
                     "date": e.get("date", ""), "label": e.get("label", ""), "note": e.get("note", ""),
                     "source": e.get("source", ""), "excluded": bool(e.get("excluded")), "values": dict(e.get("values") or {})})
    return {"columns": cols, "rows": rows}


def numeric_matrix(store, var_ids=None, include_excluded=False):
    """실험별 숫자 값. 반환: (experiment_ids, {var_id: [float|None, ...]})"""
    vars_ = store.variables()
    ids = var_ids or [v["id"] for v in sorted_variables(vars_)]
    exps = [e for e in store.experiments() if include_excluded or not e.get("excluded")]
    cols = {vid: [to_number((e.get("values") or {}).get(vid)) for e in exps] for vid in ids}
    return [e["id"] for e in exps], cols


def var_stats(store, include_excluded=False):
    """변수별 건수·평균·표준편차·최소·최대·결측."""
    eids, cols = numeric_matrix(store, include_excluded=include_excluded)
    out = {}
    n_total = len(eids)
    for vid, vals in cols.items():
        xs = [x for x in vals if x is not None]
        n = len(xs)
        st = {"n": n, "missing": n_total - n, "mean": None, "sd": None, "min": None, "max": None, "median": None}
        if n:
            m = sum(xs) / n
            st["mean"] = m
            st["min"], st["max"] = min(xs), max(xs)
            s = sorted(xs)
            st["median"] = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
            st["sd"] = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
        out[vid] = st
    return out


def project_summary(store):
    """프로젝트 카드용: 실험 수, 최근 실험일, 규격 달성 현황(가장 최근 실험 기준)."""
    exps_by = {}
    for e in store.experiments():
        exps_by.setdefault(e.get("project"), []).append(e)
    out = {}
    for p in store.projects():
        exps = exps_by.get(p["id"], [])
        dated = sorted((e for e in exps if e.get("date")), key=lambda e: e["date"])
        latest = dated[-1] if dated else (exps[-1] if exps else None)
        spec_rows = []
        for s in p.get("specs", []):
            val = (latest.get("values") or {}).get(s["var"]) if latest else None
            spec_rows.append({"var": s["var"], "spec": s["spec"], "value": val, "result": judge(s["spec"], val)})
        n_pass = sum(1 for r in spec_rows if r["result"] == "Pass")
        n_fail = sum(1 for r in spec_rows if r["result"] == "Fail")
        # 전체 실험 중 규격을 모두 만족한 실험 수
        full_pass = 0
        for e in exps:
            if p.get("specs") and all(judge(s["spec"], (e.get("values") or {}).get(s["var"])) == "Pass" for s in p["specs"]):
                full_pass += 1
        out[p["id"]] = {"n": len(exps), "n_excluded": sum(1 for e in exps if e.get("excluded")),
                        "last_date": dated[-1]["date"] if dated else "", "specs": spec_rows,
                        "n_pass": n_pass, "n_fail": n_fail, "n_spec": len(spec_rows), "full_pass": full_pass}
    return out


# ---------- 내보내기 ----------
META_HEAD = ["프로젝트", "날짜", "실험명", "메모", "출처", "제외"]


def export_rows(store, project_ids=None):
    t = table(store, project_ids)
    cols = t["columns"]
    head = META_HEAD + [f"{KIND_LABEL.get(c['kind'], '')}: {label(c)}" for c in cols]
    rows = []
    for r in t["rows"]:
        rows.append([r["project_name"], r["date"], r["label"], r["note"], r["source"], "제외" if r["excluded"] else ""]
                    + [r["values"].get(c["id"], "") for c in cols])
    return head, rows


def to_csv(head, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(head)
    w.writerows(rows)
    return ("﻿" + buf.getvalue()).encode("utf-8")


def to_xlsx(head, rows):
    import xlsx_writer
    sheet = {"name": "실험 데이터", "widths": [28, 12, 16, 28, 14, 6] + [16] * (len(head) - 6),
             "rows": [[(h, "h") for h in head]] + rows}
    return xlsx_writer.build([sheet])


# ---------- 점검 (결측·이상치·숫자 아님) ----------
def checks(store):
    """{'exp': id, 'var': id, 'type': 'outlier'|'nonnum', 'value': v, 'msg': str} 목록.
    이상치: 값이 5개 이상인 변수에서 IQR×3 밖이거나 중앙값의 8배 이상/1/8 이하."""
    out = []
    exps = store.experiments()
    for v in store.variables():
        vid = v["id"]
        pairs = [(e, (e.get("values") or {}).get(vid)) for e in exps if vid in (e.get("values") or {})]
        nums = []
        for e, val in pairs:
            n = to_number(val)
            if n is None:
                out.append({"exp": e["id"], "var": vid, "type": "nonnum", "value": val, "msg": f"'{val}'은(는) 숫자가 아니라 모델에 쓰이지 않아요"})
            else:
                nums.append((e, n))
        if len(nums) < 5:
            continue
        s = sorted(x for _, x in nums)
        n = len(s)
        q1, q3 = s[int(0.25 * (n - 1))], s[int(0.75 * (n - 1))]
        med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
        iqr = q3 - q1
        for e, x in nums:
            far = iqr > 0 and (x < q1 - 3 * iqr or x > q3 + 3 * iqr)
            digits = med > 0 and (x >= 8 * med or (x > 0 and x <= med / 8))
            if far or digits:
                out.append({"exp": e["id"], "var": vid, "type": "outlier", "value": x,
                            "msg": f"다른 값들(중앙값 {med:g}, 범위 {s[0]:g}~{s[-1]:g})과 많이 달라요. 오타인지 확인하세요"})
    return out
