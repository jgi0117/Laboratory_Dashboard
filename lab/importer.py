# -*- coding: utf-8 -*-
"""CSV/XLSX 가져오기: 파일 읽기 → 열 이름 → 변수 자동 매핑 → 실험 등록."""
import csv
import difflib
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

from .store import norm, to_number, StoreError

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
MAX_ROWS = 5000

META_PATTERNS = {
    "date": r"^(날짜|일자|date|실험일|측정일)$",
    "label": r"^(실험|실험명|실험번호|시료|시료명|샘플|sample|run|no\.?|번호|id|lot|batch|name|이름)$",
    "note": r"^(비고|메모|note|notes|remark|comment|설명)$",
    "project": r"^(프로젝트|project|과제)$",
}
KIND_HINTS = [
    ("property", r"투과율|차단율|haze|헤이즈|접착|강도|경도|저항|반사|흡수|색차|δe|점도|밀도|율$|수율|충진|기공|보이드|휨|편차|인장|신율|광택|내구|수명|효율|용량"),
    ("composition", r"함량|wt%|phr|농도|비율|배합|첨가|조성|함유|부|mol|g/l|ppm|고형분|입자|바인더|개시제|용제|모노머|올리고머|첨가제|안료|필러"),
    ("process", r"온도|시간|속도|두께|경화|압력|rpm|교반|건조|코팅|분산|전류|전압|파형|습도|거리|회전|소성|열처리|주기|횟수"),
]


# ---------- 파일 읽기 ----------
def read_csv_bytes(data):
    text = None
    for enc in ("utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise StoreError("CSV 인코딩을 읽지 못했어요 (UTF-8 또는 한글 엑셀 CSV만 지원)")
    return read_text(text)


def read_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    sample = text[:4000]
    delim = "\t" if sample.count("\t") >= sample.count(",") and "\t" in sample else ("," if "," in sample else ";")
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim)]
    rows = [r for r in rows if any(str(c).strip() for c in r)]
    if not rows:
        raise StoreError("표가 비어 있어요")
    width = max(len(r) for r in rows)
    rows = [[*r, *[""] * (width - len(r))] for r in rows]
    return [c.strip() for c in rows[0]], rows[1:MAX_ROWS + 1]


def _col_index(ref):
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx_bytes(data):
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise StoreError("xlsx 파일이 아니에요 (.xls는 지원하지 않아요)")
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
    # 첫 시트
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rid_to_target = {r.get("Id"): r.get("Target") for r in rels}
    first = wb.find("m:sheets", NS)[0]
    rid = first.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
    target = rid_to_target.get(rid, "worksheets/sheet1.xml").lstrip("/")
    path = target if target.startswith("xl/") else "xl/" + target
    sheet = ET.fromstring(z.read(path))
    date_styles = _date_styles(z)
    rows = {}
    for c in sheet.iter("{%s}c" % NS["m"]):
        ref = c.get("r")
        if not ref:
            continue
        r = int(re.search(r"\d+", ref).group(0))
        ci = _col_index(ref)
        t = c.get("t")
        v = c.find("m:v", NS)
        val = ""
        if t == "s" and v is not None:
            val = shared[int(v.text)] if v.text and v.text.isdigit() and int(v.text) < len(shared) else ""
        elif t == "inlineStr":
            val = "".join(x.text or "" for x in c.iter("{%s}t" % NS["m"]))
        elif v is not None and v.text is not None:
            val = v.text
            if c.get("s") in date_styles:
                val = _excel_date(val)
            elif t != "str":
                try:
                    f = float(val)
                    val = str(int(f)) if f.is_integer() else str(f)
                except ValueError:
                    pass
        rows.setdefault(r, {})[ci] = str(val)
    if not rows:
        raise StoreError("시트가 비어 있어요")
    order = sorted(rows)
    width = max(max(rows[r]) for r in order) + 1
    grid = [[rows[r].get(i, "") for i in range(width)] for r in order]
    grid = [r for r in grid if any(c.strip() for c in r)]
    if not grid:
        raise StoreError("시트가 비어 있어요")
    return [c.strip() for c in grid[0]], grid[1:MAX_ROWS + 1]


def _date_styles(z):
    """날짜 서식(numFmt)이 걸린 cellXfs 인덱스 집합."""
    if "xl/styles.xml" not in z.namelist():
        return set()
    root = ET.fromstring(z.read("xl/styles.xml"))
    fmts = {nf.get("numFmtId"): (nf.get("formatCode") or "") for nf in root.iter("{%s}numFmt" % NS["m"])}
    builtin = {"14", "15", "16", "17", "22"}
    out = set()
    xfs = root.find("m:cellXfs", NS)
    if xfs is None:
        return out
    for i, xf in enumerate(xfs):
        fid = xf.get("numFmtId")
        code = fmts.get(fid, "").lower()
        if fid in builtin or re.search(r"y+.*m+.*d+|d+.*m+.*y+", code):
            out.add(str(i))
    return out


def _excel_date(v):
    try:
        n = float(v)
    except ValueError:
        return v
    return (datetime(1899, 12, 30) + timedelta(days=n)).strftime("%Y-%m-%d")


def read_file(name, data):
    if str(name).lower().endswith(".xlsx"):
        return read_xlsx_bytes(data)
    return read_csv_bytes(data)


# ---------- 열 이름 해석 ----------
def split_header(h):
    """'나노입자 함량 (wt%)' → ('나노입자 함량', 'wt%'); '[μm]' 도 지원."""
    h = str(h or "").strip()
    m = re.match(r"^(.*?)\s*[\(\[（]([^()\[\]（）]{1,20})[\)\]）]\s*$", h)
    if m and m.group(1).strip():
        return m.group(1).strip(), m.group(2).strip()
    return h, ""


def strip_kind_prefix(name):
    return re.sub(r"^(조성|공정|성능|X|Y)\s*[:：]\s*", "", name, flags=re.I)


def guess_kind(name, unit=""):
    text = (name + " " + unit).lower()
    for kind, pat in KIND_HINTS:
        if re.search(pat, text, re.I):
            return kind
    return "process"


def meta_role(name):
    n = str(name or "").strip().lower()
    for role, pat in META_PATTERNS.items():
        if re.match(pat, n, re.I):
            return role
    return None


def suggest_mapping(headers, store):
    """열마다 {header, name, unit, action, var, kind, score} 제안.
    action: var(기존 변수) | new(새 변수) | date | label | note | project | skip"""
    vars_ = store.variables()
    by_norm = {}
    for v in vars_:
        by_norm.setdefault(norm(v["name"]), []).append(v)
        for a in v.get("aliases", []):
            by_norm.setdefault(norm(a), []).append(v)
    out = []
    for h in headers:
        raw_name, unit = split_header(h)
        name = strip_kind_prefix(raw_name)
        item = {"header": h, "name": name, "unit": unit, "action": "skip", "var": None, "kind": guess_kind(name, unit), "score": 0.0}
        if not name:
            out.append(item)
            continue
        role = meta_role(name)
        if role:
            item["action"] = role
            out.append(item)
            continue
        cands = by_norm.get(norm(name)) or []
        if cands:
            v = next((c for c in cands if norm(c.get("unit")) == norm(unit)), cands[0])
            item.update(action="var", var=v["id"], kind=v["kind"], score=1.0)
            if unit and norm(v.get("unit")) != norm(unit):
                item["warn"] = f"단위가 달라요: 파일 '{unit}' vs 등록 '{v.get('unit')}'"
        else:
            best, best_s = None, 0.0
            for v in vars_:
                for cand in [v["name"]] + list(v.get("aliases", [])):
                    s = difflib.SequenceMatcher(None, norm(name), norm(cand)).ratio()
                    if s > best_s:
                        best, best_s = v, s
            if best is not None and best_s >= 0.78:
                item.update(action="var", var=best["id"], kind=best["kind"], score=round(best_s, 2),
                            warn=f"비슷한 변수 '{best['name']}'에 맞췄어요. 아니면 바꿔 주세요")
            else:
                item["action"] = "new"
        out.append(item)
    return out


def preview(name, data, store, limit=15):
    headers, rows = read_file(name, data)
    return {"headers": headers, "mapping": suggest_mapping(headers, store), "rows": rows[:limit], "n": len(rows)}


def commit(store, project_id, headers, mapping, rows, source="import"):
    """매핑 확정 → 실험 등록. mapping은 preview의 항목을 (사용자가 고친 뒤) 그대로 보낸 것."""
    store.project(project_id)
    if len(mapping) != len(headers):
        raise StoreError("매핑과 열 수가 달라요")
    col_var = {}
    roles = {}
    new_vars = 0
    for i, m in enumerate(mapping):
        act = m.get("action")
        if act == "var":
            v = store.variable(m["var"])
            col_var[i] = v["id"]
            # 파일 열 이름을 별칭으로 기억해 다음 가져오기 때 자동으로 맞춘다
            nm = str(m.get("name") or "").strip()
            if nm and norm(nm) != norm(v["name"]) and norm(nm) not in {norm(a) for a in v.get("aliases", [])}:
                v.setdefault("aliases", []).append(nm)
        elif act == "new":
            nm = str(m.get("name") or "").strip()
            if not nm:
                continue
            v = store.add_variable(nm, m.get("unit") or "", m.get("kind") or "process")
            col_var[i] = v["id"]
            new_vars += 1
        elif act in ("date", "label", "note"):
            roles.setdefault(act, i)
    items = []
    for r in rows:
        values = {}
        for i, vid in col_var.items():
            if i < len(r) and str(r[i]).strip() != "":
                values[vid] = r[i]
        if not values:
            continue
        get = lambda role: str(r[roles[role]]).strip() if role in roles and roles[role] < len(r) else ""
        items.append({"project": project_id, "date": _norm_date(get("date")), "label": get("label"), "note": get("note"),
                      "source": source, "values": values})
    made = store.add_experiments(items)
    return {"added": len(made), "new_variables": new_vars, "skipped": len(rows) - len(items)}


def _norm_date(t):
    t = str(t or "").strip()
    if not t:
        return ""
    for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%Y%m%d", "%m/%d/%Y", "%d.%m.%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(t, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    n = to_number(t)
    if n is not None and 20000 < n < 80000:
        return _excel_date(n)
    return t
