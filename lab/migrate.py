# -*- coding: utf-8 -*-
"""v1 data.json(프로젝트 안에 단계별 기록) → v2(변수 사전 + 평평한 실험 표) 변환."""
import re
from collections import Counter

from .store import empty, norm, clean_value, to_number

V1_STAGES = ["제품 설계", "Lab SPL 고객 승인",
             "1차 R/A", "1차 Trial production", "1차 고객 승인",
             "2차 R/A", "2차 Trial production", "2차 고객 승인",
             "3차 R/A", "3차 Trial production", "3차 고객 승인", "양산 이관 회의"]
DESIGN = "제품 설계"
COMPOSITION_WORDS = r"함량|wt%|phr|농도|고형분|바인더|개시제|입자"
# v1 recipe 항목 이름 → 설계 실험 변수 이름 (검증 실험의 조성을 채우기 위해)
RECIPE_ALIAS = {
    "세라믹 IR 차단 나노입자 (고형분)": "나노입자 함량", "나노입자": "나노입자 함량",
    "아크릴 바인더": "바인더 함량", "바인더": "바인더 함량",
    "광개시제": "광개시제 함량",
    "건조 코팅 두께": "건조 두께",
}


def is_v1(obj):
    ps = obj.get("projects")
    return isinstance(ps, list) and (not ps or any(isinstance(p, dict) and "records" in p for p in ps)) and obj.get("version") != 2


def _kind_for_x(name, unit):
    return "composition" if re.search(COMPOSITION_WORDS, f"{name} {unit}", re.I) else "process"


def v1_to_v2(old):
    out = empty()
    vars_by_key = {}     # (norm name) -> variable

    def get_var(name, unit, kind):
        key = norm(name)
        if key in vars_by_key:
            return vars_by_key[key]
        v = {"id": f"v{len(out['variables']) + 1:04d}", "name": str(name).strip(), "unit": str(unit or "").strip(),
             "kind": kind, "aliases": [], "desc": ""}
        out["variables"].append(v)
        vars_by_key[key] = v
        return v

    # 1) 변수 사전: 모든 프로젝트의 설계 실험 x/y 열
    for p in old.get("projects", []):
        for rec in p.get("records", []):
            if rec.get("stage") != DESIGN:
                continue
            ex = rec.get("exp") or {}
            for c in ex.get("x") or []:
                if str(c.get("name", "")).strip():
                    get_var(c["name"], c.get("unit"), _kind_for_x(c["name"], c.get("unit", "")))
            for c in ex.get("y") or []:
                if str(c.get("name", "")).strip():
                    get_var(c["name"], c.get("unit"), "property")
    for src, dst in RECIPE_ALIAS.items():
        v = vars_by_key.get(norm(dst))
        if v and src not in v["aliases"]:
            v["aliases"].append(src)

    def resolve(name):
        key = norm(name)
        if key in vars_by_key:
            return vars_by_key[key]
        for v in out["variables"]:
            if key in {norm(a) for a in v["aliases"]}:
                return v
        return None

    # 2) 프로젝트·실험
    n_exp = 0
    for i, p in enumerate(old.get("projects", []), 1):
        pid = f"p{i:04d}"
        name = str(p.get("name") or "").strip() or f"프로젝트 {i}"
        product = ""
        m = re.match(r"^(\[예시\]\s*)?(차량용 윈도우 필름)\s*(.*)$", name)
        if m:
            product = m.group(2)
            name = (m.group(1) or "") + m.group(3)
        pos = p.get("pos") if isinstance(p.get("pos"), int) else 0
        stage = V1_STAGES[pos] if 0 <= pos < len(V1_STAGES) else ""
        status = "완료" if pos >= len(V1_STAGES) - 1 else "진행"
        # 규격: 각 평가 항목에서 가장 많이 쓰인 spec
        spec_votes = {}
        for rec in p.get("records", []):
            for e in rec.get("evals") or []:
                item, spec = str(e.get("item") or "").strip(), str(e.get("spec") or "").strip()
                if item and spec:
                    spec_votes.setdefault(item, Counter())[spec] += 1
        specs = []
        for item, votes in spec_votes.items():
            v = resolve(item)
            if v is None:
                v = get_var(item, next((e.get("unit") for rec in p.get("records", []) for e in rec.get("evals") or [] if e.get("item") == item), ""), "property")
            if v["id"] not in {s["var"] for s in specs}:
                specs.append({"var": v["id"], "spec": votes.most_common(1)[0][0]})
        notes = [str(p.get("note") or "").strip()]
        exps = []
        for rec in sorted(p.get("records", []), key=lambda r: (r.get("attempt", 1), V1_STAGES.index(r["stage"]) if r.get("stage") in V1_STAGES else 99)):
            stage_name, attempt = rec.get("stage"), rec.get("attempt", 1)
            if stage_name == DESIGN:
                ex = rec.get("exp") or {}
                xcols = {c["id"]: resolve(c["name"]) for c in ex.get("x") or [] if c.get("name")}
                ycols = {c["id"]: resolve(c["name"]) for c in ex.get("y") or [] if c.get("name")}
                if str(rec.get("note") or "").strip():
                    notes.append(f"[설계 메모 (시도 {attempt})] {rec['note'].strip()}")
                struct = rec.get("structure") or []
                if struct:
                    parts = []
                    for l in struct:
                        bits = [str(l.get("layer") or "").strip(), str(l.get("material") or "").strip()]
                        if str(l.get("thickness") or "").strip():
                            bits.append(str(l.get("thickness")).strip() + str(l.get("unit") or "").strip())
                        parts.append(" ".join(b for b in bits if b))
                    notes.append("[제품 구조] " + " / ".join(x for x in parts if x))
                for n, run in enumerate(ex.get("runs") or [], 1):
                    values = {}
                    for cid, v in xcols.items():
                        if v is not None:
                            cv = clean_value((run.get("x") or {}).get(cid))
                            if cv is not None:
                                values[v["id"]] = cv
                    for cid, v in ycols.items():
                        if v is not None:
                            cv = clean_value((run.get("y") or {}).get(cid))
                            if cv is not None:
                                values[v["id"]] = cv
                    if not values:
                        continue
                    n_exp += 1
                    exps.append({"id": f"e{n_exp:04d}", "project": pid, "date": str(run.get("date") or ""),
                                 "label": f"설계{attempt}-{n:02d}" + (" ★" if run.get("star") else ""),
                                 "note": str(run.get("note") or ""), "source": "migrated:설계",
                                 "values": values, "excluded": bool(run.get("excl"))})
            else:
                evals = [e for e in rec.get("evals") or [] if str(e.get("item") or "").strip()]
                if not evals:
                    continue
                values = {}
                for r in rec.get("recipe") or []:
                    v = resolve(r.get("item"))
                    if v is not None and v["kind"] != "property":
                        cv = clean_value(r.get("value"))
                        if cv is not None and to_number(cv) is not None:
                            values[v["id"]] = cv
                for e in evals:
                    v = resolve(e.get("item"))
                    if v is None:
                        v = get_var(e["item"], e.get("unit"), "property")
                    cv = clean_value(e.get("value"))
                    if cv is not None:
                        values[v["id"]] = cv
                if not values:
                    continue
                n_exp += 1
                exps.append({"id": f"e{n_exp:04d}", "project": pid, "date": "",
                             "label": f"{stage_name} (시도 {attempt})", "note": str(rec.get("note") or "").strip(),
                             "source": "migrated:검증", "values": values, "excluded": False})
        dates = [e["date"] for e in exps if e["date"]]
        out["projects"].append({"id": pid, "name": name, "product": product, "owner": "", "status": status, "stage": stage,
                                "note": "\n".join(x for x in notes if x), "specs": specs,
                                "created": min(dates) if dates else str(p.get("updated") or ""),
                                "updated": str(p.get("updated") or "")})
        out["experiments"].extend(exps)
    return out


if __name__ == "__main__":
    import json
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "data.json"
    obj = json.load(open(src, encoding="utf-8"))
    new = v1_to_v2(obj)
    print(json.dumps({"variables": len(new["variables"]), "projects": len(new["projects"]), "experiments": len(new["experiments"])}, ensure_ascii=False))
