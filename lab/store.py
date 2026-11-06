# -*- coding: utf-8 -*-
"""data.json 읽기·쓰기와 데이터 모델 (version 2).

{
  "version": 2,
  "variables":   [{id, name, unit, kind, aliases, desc}],       kind: composition | process | property
  "projects":    [{id, name, product, owner, status, stage, note, specs:[{var, spec}], created, updated}],
  "experiments": [{id, project, date, label, note, source, values:{var_id: 값}, excluded}]
}
"""
import json
import os
import re
import tempfile
import threading
from datetime import date
from pathlib import Path

KINDS = ("composition", "process", "property")
KIND_LABEL = {"composition": "조성", "process": "공정", "property": "성능"}
STATUSES = ("진행", "보류", "완료")


def today():
    return date.today().isoformat()


def norm(s):
    """이름 비교용 정규화: 소문자, 공백 제거, 비슷한 기호 통일."""
    s = str(s or "").strip().lower()
    s = s.replace("µ", "μ").replace("℃", "°c").replace("˚", "°").replace("（", "(").replace("）", ")")
    s = re.sub(r"[\s_\-·,/]+", "", s)
    return s


def to_number(v):
    """숫자로 읽을 수 있으면 float, 아니면 None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    t = str(v or "").strip().replace(",", "")
    t = re.sub(r"^[≈~약]\s*", "", t)
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def clean_value(v):
    """저장용 값: 숫자면 float, 빈 값이면 None, 그 밖은 문자열."""
    if v is None:
        return None
    if isinstance(v, bool):
        return str(v)
    n = to_number(v)
    if n is not None:
        return n
    t = str(v).strip()
    return t or None


def empty():
    return {"version": 2, "variables": [], "projects": [], "experiments": []}


class StoreError(ValueError):
    pass


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.data = self._load()

    # ---------- 파일 ----------
    def _load(self):
        try:
            obj = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return empty()
        if obj.get("version") != 2:
            from . import migrate
            if migrate.is_v1(obj):
                backup = self.path.with_name(self.path.stem + ".v1.backup.json")
                if not backup.exists():
                    backup.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
                obj = migrate.v1_to_v2(obj)
                self._write(obj)
            else:
                obj = empty()
        for k in ("variables", "projects", "experiments"):
            obj.setdefault(k, [])
        return obj

    def _write(self, obj):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def save(self):
        with self.lock:
            self._write(self.data)

    def reload(self):
        with self.lock:
            self.data = self._load()

    # ---------- 공통 ----------
    def new_id(self, prefix):
        coll = {"v": "variables", "p": "projects", "e": "experiments"}[prefix]
        best = 0
        for o in self.data[coll]:
            m = re.match(prefix + r"(\d+)$", str(o.get("id", "")))
            if m:
                best = max(best, int(m.group(1)))
        return f"{prefix}{best + 1:04d}"

    def _get(self, coll, oid):
        for o in self.data[coll]:
            if o.get("id") == oid:
                return o
        raise StoreError(f"없는 항목: {oid}")

    # ---------- 변수 ----------
    def variables(self):
        return list(self.data["variables"])

    def variable(self, vid):
        return self._get("variables", vid)

    def find_variable(self, name, unit=None):
        """이름(또는 별칭)이 같은 변수. unit이 주어지면 단위까지 같은 것을 우선, 없으면 이름만 같은 것."""
        n = norm(name)
        if not n:
            return None
        same_name = [v for v in self.data["variables"]
                     if norm(v["name"]) == n or n in {norm(a) for a in v.get("aliases", [])}]
        if not same_name:
            return None
        if unit is not None:
            u = norm(unit)
            for v in same_name:
                if norm(v.get("unit")) == u:
                    return v
        return same_name[0]

    def add_variable(self, name, unit="", kind="process", aliases=None, desc=""):
        name = str(name or "").strip()
        if not name:
            raise StoreError("변수 이름이 비어 있어요")
        if kind not in KINDS:
            raise StoreError(f"변수 구분은 {KINDS} 중 하나")
        with self.lock:
            found = self.find_variable(name, unit)
            if found and norm(found.get("unit")) == norm(unit):
                return found
            v = {"id": self.new_id("v"), "name": name, "unit": str(unit or "").strip(), "kind": kind,
                 "aliases": [a for a in (aliases or []) if str(a).strip()], "desc": str(desc or "")}
            self.data["variables"].append(v)
            return v

    def update_variable(self, vid, patch):
        with self.lock:
            v = self.variable(vid)
            for k in ("name", "unit", "desc"):
                if k in patch:
                    v[k] = str(patch[k] or "").strip()
            if "kind" in patch:
                if patch["kind"] not in KINDS:
                    raise StoreError("잘못된 변수 구분")
                v["kind"] = patch["kind"]
            if "aliases" in patch:
                v["aliases"] = [str(a).strip() for a in patch["aliases"] if str(a).strip()]
            if not v["name"]:
                raise StoreError("변수 이름이 비어 있어요")
            return v

    def variable_usage(self, vid):
        n_exp = sum(1 for e in self.data["experiments"] if vid in (e.get("values") or {}))
        n_spec = sum(1 for p in self.data["projects"] for s in p.get("specs", []) if s.get("var") == vid)
        return n_exp, n_spec

    def delete_variable(self, vid, force=False):
        with self.lock:
            v = self.variable(vid)
            n_exp, n_spec = self.variable_usage(vid)
            if (n_exp or n_spec) and not force:
                raise StoreError(f"'{v['name']}'은(는) 실험 {n_exp}건·규격 {n_spec}건에서 쓰고 있어서 지울 수 없어요")
            for e in self.data["experiments"]:
                (e.get("values") or {}).pop(vid, None)
            for p in self.data["projects"]:
                p["specs"] = [s for s in p.get("specs", []) if s.get("var") != vid]
            self.data["variables"] = [x for x in self.data["variables"] if x["id"] != vid]

    def merge_variables(self, src, dst):
        """src 변수를 dst로 합친다. 값이 둘 다 있으면 dst 값을 남긴다."""
        with self.lock:
            if src == dst:
                raise StoreError("같은 변수예요")
            s, d = self.variable(src), self.variable(dst)
            for e in self.data["experiments"]:
                vals = e.get("values") or {}
                if src in vals:
                    vals.setdefault(dst, vals[src])
                    del vals[src]
            for p in self.data["projects"]:
                for sp in p.get("specs", []):
                    if sp.get("var") == src:
                        sp["var"] = dst
                seen, out = set(), []
                for sp in p.get("specs", []):
                    if sp["var"] not in seen:
                        seen.add(sp["var"])
                        out.append(sp)
                p["specs"] = out
            d.setdefault("aliases", [])
            for a in [s["name"]] + list(s.get("aliases", [])):
                if a and norm(a) != norm(d["name"]) and norm(a) not in {norm(x) for x in d["aliases"]}:
                    d["aliases"].append(a)
            self.data["variables"] = [x for x in self.data["variables"] if x["id"] != src]
            return d

    # ---------- 프로젝트 ----------
    def projects(self):
        return list(self.data["projects"])

    def project(self, pid):
        return self._get("projects", pid)

    def add_project(self, fields):
        with self.lock:
            p = {"id": self.new_id("p"), "name": "", "product": "", "owner": "", "status": "진행", "stage": "",
                 "note": "", "specs": [], "created": today(), "updated": today()}
            self._apply_project(p, fields)
            self.data["projects"].append(p)
            return p

    def _apply_project(self, p, fields):
        for k in ("name", "product", "owner", "stage", "note", "created"):
            if k in fields:
                p[k] = str(fields[k] or "").strip()
        if "status" in fields:
            if fields["status"] not in STATUSES:
                raise StoreError(f"상태는 {STATUSES} 중 하나")
            p["status"] = fields["status"]
        if "specs" in fields:
            specs = []
            for s in fields["specs"] or []:
                var, spec = s.get("var"), str(s.get("spec") or "").strip()
                if var and spec:
                    self.variable(var)
                    specs.append({"var": var, "spec": spec})
            p["specs"] = specs
        if not p["name"]:
            raise StoreError("프로젝트 이름이 비어 있어요")
        p["updated"] = today()

    def update_project(self, pid, patch):
        with self.lock:
            p = self.project(pid)
            self._apply_project(p, patch)
            return p

    def delete_project(self, pid):
        with self.lock:
            self.project(pid)
            self.data["experiments"] = [e for e in self.data["experiments"] if e.get("project") != pid]
            self.data["projects"] = [p for p in self.data["projects"] if p["id"] != pid]

    # ---------- 실험 ----------
    def experiments(self, project=None):
        return [e for e in self.data["experiments"] if project is None or e.get("project") == project]

    def experiment(self, eid):
        return self._get("experiments", eid)

    def _clean_values(self, values):
        out = {}
        for k, v in (values or {}).items():
            self.variable(k)
            cv = clean_value(v)
            if cv is not None:
                out[k] = cv
        return out

    def add_experiments(self, items):
        with self.lock:
            made = []
            for it in items:
                pid = it.get("project")
                self.project(pid)
                e = {"id": self.new_id("e"), "project": pid, "date": str(it.get("date") or "").strip(),
                     "label": str(it.get("label") or "").strip(), "note": str(it.get("note") or "").strip(),
                     "source": str(it.get("source") or "manual"), "values": self._clean_values(it.get("values")),
                     "excluded": bool(it.get("excluded", False))}
                self.data["experiments"].append(e)
                self.project(pid)["updated"] = today()
                made.append(e)
            return made

    def update_experiment(self, eid, patch):
        with self.lock:
            e = self.experiment(eid)
            for k in ("date", "label", "note", "source"):
                if k in patch:
                    e[k] = str(patch[k] or "").strip()
            if "project" in patch:
                self.project(patch["project"])
                e["project"] = patch["project"]
            if "excluded" in patch:
                e["excluded"] = bool(patch["excluded"])
            if "values" in patch:          # 전체 교체
                e["values"] = self._clean_values(patch["values"])
            if "set" in patch:             # 일부 칸만 수정 (None/빈 값이면 지움)
                for k, v in (patch["set"] or {}).items():
                    self.variable(k)
                    cv = clean_value(v)
                    if cv is None:
                        e["values"].pop(k, None)
                    else:
                        e["values"][k] = cv
            self.project(e["project"])["updated"] = today()
            return e

    def delete_experiment(self, eid):
        with self.lock:
            self.experiment(eid)
            self.data["experiments"] = [e for e in self.data["experiments"] if e["id"] != eid]

    # ---------- 통계 ----------
    def summary(self):
        return {"variables": len(self.data["variables"]), "projects": len(self.data["projects"]),
                "experiments": len(self.data["experiments"]),
                "excluded": sum(1 for e in self.data["experiments"] if e.get("excluded"))}
