# -*- coding: utf-8 -*-
"""실험 노트 점검기 (로컬 서버).

실행:  python server.py   → http://localhost:8840/
AI는 이 PC의 Claude Code(claude -p)를 호출한다. API 키는 쓰지 않고, Claude Code에 로그인된 내 계정을 그대로 쓴다.
통계·이상치·그래프는 AI가 아니라 브라우저의 코드가 계산한다. AI는 메모를 표로 옮기는 일과 의미 점검만 맡는다.

원칙: 누적 데이터는 이 폴더의 data.json에만 저장하고(요청 내용은 로그에 남기지 않음), 이 PC의 브라우저에서 온 요청만 받는다.
"""
import base64
import csv
import datetime
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import prompts

BASE = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8840"))
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
MODEL = os.environ.get("CLAUDE_MODEL", "")
JOBS, JOBS_LOCK = {}, threading.Lock()
DATA = BASE / "data.json"          # 누적 데이터 (이 PC에만 저장, git에는 올리지 않음)
DATA_LOCK = threading.Lock()
MAX_TEXT = 40000
UNC_RE = re.compile(r"정도|약\s?\d|쯤|듯|\?|대략|추정|~")      # 불확실하게 적은 표현
UNIT_ALIAS = {"um": "μm", "µm": "μm", "μm": "μm", "micron": "μm", "마이크로": "μm", "도": "℃", "°c": "℃", "℃": "℃", "c": "℃",
              "분": "min", "min": "min", "초": "s", "sec": "s", "시간": "h", "hr": "h", "asd": "ASD"}


# ---------- Claude Code ----------
def find_claude():
    if os.environ.get("CLAUDE_BIN"):
        return os.environ["CLAUDE_BIN"]
    found = shutil.which("claude")
    if found:
        return found
    hits = sorted(glob.glob(os.path.join(os.environ.get("APPDATA", ""), "Claude", "claude-code", "*", "*", "claude.exe")))
    return hits[-1] if hits else None


def run_claude(prompt, timeout=600):
    exe = find_claude()
    if not exe:
        return None, "claude 실행 파일을 찾지 못했어요. CLAUDE_BIN 환경변수에 경로를 지정해 주세요."
    cmd = [exe, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence"] + (["--model", MODEL] if MODEL else [])
    try:
        p = subprocess.run(cmd, input=prompt.encode("utf-8"), capture_output=True, timeout=timeout, creationflags=NO_WINDOW, cwd=str(BASE))
    except subprocess.TimeoutExpired:
        return None, "AI 응답이 너무 오래 걸려요. 다시 시도해 주세요."
    try:
        data = json.loads(p.stdout.decode("utf-8", "replace").strip())
    except json.JSONDecodeError:
        return None, "Claude Code 응답을 읽지 못했어요."
    if data.get("is_error"):
        return None, "NOT_LOGGED_IN" if "login" in str(data.get("result", "")).lower() else "Claude Code 오류가 났어요."
    return str(data.get("result", "")), None


class BadShape(Exception):
    pass


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("JSON 없음")
    return json.loads(m.group(0))


def call_json(prompt, validate):
    """JSON을 받아 모양을 검사한다. 틀리면 한 번만 다시 요청하고, 그래도 틀리면 오류를 그대로 돌려준다."""
    for attempt in range(2):
        text, err = run_claude(prompt if attempt == 0 else prompt + "\n\n[주의] 직전 답이 형식에 맞지 않았다. 지정한 JSON 형식을 정확히 지켜서 JSON 하나만 다시 출력해라.")
        if err:
            return None, err
        try:
            return validate(extract_json(text)), None
        except (ValueError, json.JSONDecodeError, BadShape, TypeError, AttributeError):
            continue
    return None, "AI 답이 형식에 맞지 않았어요. 다시 시도해 주세요."


# ---------- 코드로 하는 검증 ----------
def norm_quote(s):
    return re.sub(r"[\s\W_]+", "", str(s or "")).lower()


def norm_unit(u):
    u = str(u or "").strip()
    return UNIT_ALIAS.get(u.lower(), u)


NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def clean_num(v):
    """숫자 열의 값: 숫자로 읽히면 숫자 문자열, 아니면 None(원문은 따로 보관)."""
    s = str(v).replace(",", "").strip()
    return s if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s) else None


def validate_structure(ctx):
    hay = norm_quote(ctx.get("memo", "") + "\n" + ctx.get("table", ""))

    def v(res):
        cols_in, rows_in = res.get("columns"), res.get("rows")
        if not isinstance(cols_in, list) or not isinstance(rows_in, list) or not cols_in:
            raise BadShape
        cols, seen = [], set()
        for c in cols_in[:30]:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            cols.append({"id": f"c{len(cols)+1}", "name": name, "unit": norm_unit(c.get("unit")),
                         "role": c.get("role") if c.get("role") in ("condition", "result", "info") else "info",
                         "kind": "number" if c.get("kind") == "number" else "text"})
        byname = {c["name"]: c for c in cols}
        rows = []
        for i, r in enumerate(rows_in[:200], 1):
            if not isinstance(r, dict):
                continue
            cells = {}
            raw = r.get("cells") if isinstance(r.get("cells"), dict) else {}
            for name, c in byname.items():
                x = raw.get(name)
                x = x if isinstance(x, dict) else ({"value": x} if x not in (None, "") else {})
                val = x.get("value")
                val = None if val is None or str(val).strip() == "" else str(val).strip()
                q = str(x.get("quote") or "").strip()
                cell = {"v": val, "q": q, "qok": (bool(q) and norm_quote(q) in hay) if val is not None else True,
                        "unc": (x.get("certain") is False) or bool(UNC_RE.search(q)) if val is not None else False}
                if c["kind"] == "number" and val is not None:
                    n = clean_num(val)
                    if n is None:
                        cell["raw"] = val
                        cell["v"] = None
                        cell["nonnum"] = True
                    else:
                        cell["v"] = n
                cu = norm_unit(x.get("unit"))
                if val is not None and cu and c["unit"] and cu != c["unit"]:
                    cell["unit"] = cu               # 열 단위와 다르게 적힌 셀 (환산하지 않고 표시만)
                alt = x.get("alt")
                if isinstance(alt, dict) and str(alt.get("value") or "").strip():
                    cell["alt"] = str(alt["value"]).strip()
                    cell["altq"] = str(alt.get("quote") or "")
                cells[c["id"]] = cell
            rows.append({"id": f"r{i}", "label": str(r.get("label") or f"{i}번"), "date": str(r.get("date") or ""), "cells": cells})
        if not rows:
            raise BadShape
        return {"columns": cols, "rows": rows,
                "notes": [str(x) for x in (res.get("notes") or [])][:6], "questions": [str(x) for x in (res.get("questions") or [])][:6]}
    return v


def validate_review(res):
    f = res.get("findings")
    if not isinstance(f, list):
        raise BadShape
    return {"findings": [{**{k: str(x.get(k, "")) for k in ("row", "column", "issue", "why", "suggestion")},
                          "severity": x.get("severity") if x.get("severity") in ("high", "mid", "low") else "mid"}
                         for x in f if isinstance(x, dict) and str(x.get("issue", "")).strip()][:30],
            "questions": [str(x) for x in (res.get("questions") or [])][:6]}


# ---------- 표 파일 읽기 (xlsx / csv) ----------
MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RELNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
NUM_CELL = re.compile(r"^\s*(?:약\s*)?([-+]?\d[\d,]*(?:\.\d+)?|[-+]?\.\d+)\s*([A-Za-zμµ℃°%/·][A-Za-z0-9/·]*)?\s*(?:정도|쯤)?\s*$")
EMPTY = {"", "-", "–", "—", "n/a", "na", "null", "none", "미측정", "?"}   # "없음"은 보이드 없음 같은 실제 값이라 빈칸으로 보지 않는다
LABEL_KW = ("실험", "번호", "no", "id", "이름", "시료", "샘플", "sample", "label", "run", "lot", "시편")
DATE_KW = ("날짜", "일자", "date", "일시", "측정일", "실험일", "시험일", "작성일")
RESULT_KW = ("두께", "강도", "거칠기", "ra", "수율", "결과", "저항", "효율", "접착", "투과", "반사", "경도", "무게", "중량", "판정", "pass", "fail", "합격")
INFO_KW = ("비고", "메모", "note", "특이", "코멘트", "comment", "remark", "담당", "작성")
HEAD_UNIT = re.compile(r"^(.*?)\s*[\(\[（]([^\)\]）]*)[\)\]）]\s*$")


def _colidx(ref):
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def _fnum(s):
    try:
        return "%.12g" % float(s)
    except (TypeError, ValueError):
        return s or ""


def read_xlsx(b):
    z = zipfile.ZipFile(io.BytesIO(b))
    if sum(i.file_size for i in z.infolist()) > 60_000_000:
        raise ValueError("파일이 너무 커요")
    names = z.namelist()
    ss = []
    if "xl/sharedStrings.xml" in names:
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(MAIN + "si"):
            ss.append("".join(t.text or "" for t in si.iter(MAIN + "t")))
    first = ET.fromstring(z.read("xl/workbook.xml")).find(MAIN + "sheets")[0]
    rid, target = first.get(RELNS + "id"), None
    for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")):
        if r.get("Id") == rid:
            target = r.get("Target")
    path = target[1:] if target.startswith("/") else "xl/" + target
    out = []
    for row in ET.fromstring(z.read(path)).iter(MAIN + "row"):
        vals = {}
        for c in row.findall(MAIN + "c"):
            t, v = c.get("t"), c.find(MAIN + "v")
            if t == "inlineStr":
                txt = "".join(x.text or "" for x in c.iter(MAIN + "t"))
            elif v is None:
                continue
            elif t == "s":
                txt = ss[int(v.text)]
            elif t in ("str", "e"):
                txt = v.text or ""
            elif t == "b":
                txt = "TRUE" if v.text == "1" else "FALSE"
            else:
                txt = _fnum(v.text)
            vals[_colidx(c.get("r"))] = txt
        if vals:
            out.append([vals.get(i, "") for i in range(max(vals) + 1)])
    return out


def read_text_table(raw):
    if isinstance(raw, bytes):
        for enc in ("utf-8-sig", "cp949"):
            try:
                raw = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raw = raw.decode("utf-8", "replace")
    first = raw.splitlines()[0] if raw.strip() else ""
    delim = max(("\t", ",", ";"), key=lambda d: first.count(d))
    return [r for r in csv.reader(io.StringIO(raw), delimiter=delim)]


def parse_cell(raw):
    """숫자 칸: 숫자·단위·불확실 표기를 분리한다. 값을 계산하거나 환산하지 않는다."""
    m = NUM_CELL.match(raw)
    if not m:
        return None
    return {"v": m.group(1).replace(",", ""), "unit": norm_unit(m.group(2)), "unc": bool(re.search(r"약|정도|쯤", raw))}


def tabulate(grid):
    grid = [r for r in grid if any(str(x).strip() for x in r)]
    if len(grid) < 2:
        raise ValueError("머리글 행과 데이터 행이 필요해요")
    head = [str(x).strip() for x in grid[0]]
    while head and not head[-1]:
        head.pop()
    n = len(head)
    if not n or n > 60:
        raise ValueError("열이 없거나 너무 많아요")
    body = [([str(x).strip() for x in r] + [""] * n)[:n] for r in grid[1:2001]]
    li = di = None
    cols, meta, used = [], [], set()
    for j, h in enumerate(head):
        name, unit = (h or f"열{j+1}"), ""
        m = HEAD_UNIT.match(name)
        if m and m.group(1).strip():
            name, unit = m.group(1).strip(), norm_unit(m.group(2))
        low = name.lower()
        if li is None and any(k == low or k in low for k in LABEL_KW) and not any(k in low for k in RESULT_KW):
            li = j
            continue
        if di is None and any(k in low for k in DATE_KW):
            di = j
            continue
        while name in used:
            name += "'"
        used.add(name)
        vals = [r[j] for r in body if r[j].strip().lower() not in EMPTY]
        ok = [p for p in (parse_cell(v) for v in vals) if p]
        kind = "number" if vals and len(ok) >= 0.6 * len(vals) else "text"
        if kind == "number" and not unit:
            us = [p["unit"] for p in ok if p["unit"]]
            unit = max(set(us), key=us.count) if us else ""
        long_text = kind == "text" and vals and sum(len(v) for v in vals) / len(vals) > 20
        if any(k in low for k in INFO_KW) or long_text:
            role = "info"
        elif any(k in low for k in RESULT_KW):
            role = "result"
        else:
            role = "condition"
        cols.append({"name": name, "unit": unit, "kind": kind, "role": role})
        meta.append(j)
    rows = []
    for i, r in enumerate(body, 1):
        cells = {}
        for c, j in zip(cols, meta):
            raw = r[j].strip()
            if raw.lower() in EMPTY:
                cells[c["name"]] = {"v": None}
            elif c["kind"] == "number":
                p = parse_cell(raw)
                if not p:
                    cells[c["name"]] = {"v": None, "raw": raw, "nonnum": True}
                else:
                    x = {"v": p["v"]}
                    if p["unit"] and c["unit"] and p["unit"] != c["unit"]:
                        x["unit"] = p["unit"]
                    if p["unc"]:
                        x["unc"] = True
                    cells[c["name"]] = x
            else:
                cells[c["name"]] = {"v": raw}
        date = r[di] if di is not None else ""
        if re.fullmatch(r"\d{5}(\.\d+)?", date) and 30000 < float(date) < 60000:     # 엑셀 날짜 번호
            date = (datetime.date(1899, 12, 30) + datetime.timedelta(days=int(float(date)))).isoformat()
        rows.append({"label": r[li] if li is not None and r[li] else f"{i}행", "date": date, "cells": cells})
    return {"cols": cols, "rows": rows, "truncated": len(grid) - 1 > 2000}


def import_table(body):
    name = str(body.get("name") or "").lower()
    if name.endswith(".xls"):
        raise ValueError("옛 형식(.xls)은 읽지 못해요. 엑셀에서 .xlsx나 .csv로 저장해 올려 주세요")
    if body.get("b64"):
        raw = base64.b64decode(body["b64"])
        if len(raw) > 10_000_000:
            raise ValueError("파일이 너무 커요")
        grid = read_xlsx(raw) if name.endswith(".xlsx") else read_text_table(raw)
    else:
        grid = read_text_table(str(body.get("text") or ""))
    return tabulate(grid)


def load_data():
    with DATA_LOCK:
        try:
            return json.loads(DATA.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}


def save_data(d):
    if not isinstance(d, dict) or not isinstance(d.get("cols"), list) or not isinstance(d.get("rows"), list):
        raise ValueError("bad data")
    keep = {k: d.get(k) for k in ("schema", "cols", "rows", "rev", "ex", "batches", "seq")}
    tmp = DATA.with_suffix(".tmp")
    with DATA_LOCK:
        tmp.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, DATA)


# ---------- 틀만 있는 엑셀 만들기 ----------
def _colname(i):
    out, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(65 + r) + out
    return out


def build_template(cols):
    from xml.sax.saxutils import escape
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    style = {"condition": 1, "result": 2, "info": 3}
    rolename = {"condition": "입력", "result": "출력", "info": "기타"}
    cols = [c for c in cols[:60] if isinstance(c, dict) and str(c.get("name", "")).strip()]
    head = [("실험 이름", "info"), ("날짜", "info")] + [
        ((f"{str(c['name'])[:40]} ({str(c.get('unit'))[:12]})" if c.get("unit") else str(c["name"])[:40]), c.get("role", "info")) for c in cols]

    def cell(ref, txt, st):
        return f'<c r="{ref}" s="{st}" t="inlineStr"><is><t xml:space="preserve">{escape(str(txt))}</t></is></c>'

    row1 = "".join(cell(f"{_colname(i)}1", txt, style.get(role, 3)) for i, (txt, role) in enumerate(head))
    widths = "".join(f'<col min="{i+1}" max="{i+1}" width="{min(30, max(10, len(txt) * 2 + 3))}" customWidth="1"/>' for i, (txt, _r) in enumerate(head))
    sheet1 = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet {ns}><sheetViews><sheetView workbookViewId="0">'
              '<pane xSplit="2" ySplit="1" topLeftCell="C2" activePane="bottomRight" state="frozen"/></sheetView></sheetViews>'
              f'<cols>{widths}</cols><sheetData><row r="1" ht="36" customHeight="1">{row1}</row></sheetData></worksheet>')
    guide = [("항목", "구분", "단위", "적는 방법")] + [
        (str(c["name"]), rolename.get(c.get("role"), "기타"), str(c.get("unit") or "-"), "숫자만 적기 (단위는 머리글에 있음)" if c.get("kind") == "number" else "글자 그대로 적기") for c in cols]
    rows2 = "".join(
        f'<row r="{r}">' + "".join(cell(f"{_colname(j)}{r}", v, 4 if r == 1 else 0) for j, v in enumerate(vals)) + "</row>"
        for r, vals in enumerate(guide, 1))
    sheet2 = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet {ns}><cols><col min="1" max="1" width="22" customWidth="1"/>'
              f'<col min="2" max="3" width="10" customWidth="1"/><col min="4" max="4" width="40" customWidth="1"/></cols><sheetData>{rows2}</sheetData></worksheet>')

    def fill(rgb):
        return f'<fill><patternFill patternType="solid"><fgColor rgb="{rgb}"/><bgColor indexed="64"/></patternFill></fill>'
    bd = '<border><left style="thin"><color rgb="FF999999"/></left><right style="thin"><color rgb="FF999999"/></right><top style="thin"><color rgb="FF999999"/></top><bottom style="thin"><color rgb="FF999999"/></bottom><diagonal/></border>'
    al = '<alignment horizontal="center" vertical="center" wrapText="1"/>'

    def xf(fill_id, border_id, align):
        return f'<xf numFmtId="0" fontId="1" fillId="{fill_id}" borderId="{border_id}" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1">{align}</xf>'
    styles = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet {ns}>'
              '<fonts count="2"><font><sz val="11"/><name val="맑은 고딕"/></font><font><b/><sz val="11"/><name val="맑은 고딕"/></font></fonts>'
              '<fills count="5"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>'
              + fill("FFD9EFEB") + fill("FFFDE3D0") + fill("FFE5E7EB") + '</fills>'
              f'<borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border>{bd}</borders>'
              '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
              '<cellXfs count="5"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
              + xf(2, 1, al) + xf(3, 1, al) + xf(4, 1, al) + xf(0, 1, "") + '</cellXfs></styleSheet>')
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
          '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
          '<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
          '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    wb = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook {ns} xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
          '<sheets><sheet name="실험 기록" sheetId="1" r:id="rId1"/><sheet name="작성 안내" sheetId="2" r:id="rId2"/></sheets></workbook>')
    wbrels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
              '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
              '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>'
              '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("xl/workbook.xml", wb)
        z.writestr("xl/_rels/workbook.xml.rels", wbrels)
        z.writestr("xl/styles.xml", styles)
        z.writestr("xl/worksheets/sheet1.xml", sheet1)
        z.writestr("xl/worksheets/sheet2.xml", sheet2)
    return buf.getvalue()


def clip(s, n=MAX_TEXT):
    return str(s or "")[:n]


def run_job(job_id, stage, data):
    try:
        d = {"memo": clip(data.get("memo")), "table": clip(data.get("table"), 15000)}
        if isinstance(data.get("schema"), list):
            d["schema"] = [{k: str(x.get(k, ""))[:40] for k in ("name", "unit", "role")} for x in data["schema"][:40] if isinstance(x, dict)]
        if stage == "structure":
            if not d["memo"].strip() and not d["table"].strip():
                raise ValueError("실험 메모를 먼저 넣어 주세요")
            result, err = call_json(prompts.structure_prompt(d), validate_structure(d))
        else:
            d.update(columns=data.get("columns") or [], rows=data.get("rows") or [], found=data.get("found") or [])
            result, err = call_json(prompts.review_prompt(d), validate_review)
    except ValueError as e:
        result, err = None, str(e)
    except Exception:   # noqa: BLE001  내용이 로그에 남지 않도록 고정 문구만 쓴다
        result, err = None, "서버 오류가 났어요. 다시 시도해 주세요."
    with JOBS_LOCK:
        JOBS[job_id].update({"status": "error" if err else "done", "error": err, "result": result, "ended": time.time()})


def start_job(stage, data):
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        for k in [k for k, v in JOBS.items() if time.time() - v["started"] > 3600]:
            del JOBS[k]
        JOBS[job_id] = {"status": "running", "started": time.time(), "result": None, "error": None}
    threading.Thread(target=run_job, args=(job_id, stage, data), daemon=True).start()
    return job_id


_auth = {"t": 0, "v": None}


def auth_status():
    if time.time() - _auth["t"] < 15 and _auth["v"] is not None:
        return _auth["v"]
    exe, v = find_claude(), {"claude": bool(find_claude()), "loggedIn": False}
    if exe:
        try:
            p = subprocess.run([exe, "auth", "status"], capture_output=True, timeout=20, creationflags=NO_WINDOW)
            v["loggedIn"] = bool(json.loads(p.stdout.decode("utf-8", "replace")).get("loggedIn"))
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
            pass
    _auth.update(t=time.time(), v=v)
    return v


def open_login_window():
    exe = find_claude()
    if not exe or os.name != "nt":
        return False
    subprocess.Popen(["cmd", "/c", "start", "Claude 로그인", "cmd", "/k", f'"{exe}" auth login'])
    _auth["t"] = 0
    return True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass        # 요청 내용을 로그로 남기지 않는다

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _local_only(self):
        host, origin = self.headers.get("Host", ""), self.headers.get("Origin", "")
        ok = (re.match(r"^(localhost|127\.0\.0\.1)(:\d+)?$", host) and self.client_address[0] in ("127.0.0.1", "::1")
              and (not origin or origin == "http://" + host))
        if not ok:
            self._send(403, {"error": "허용되지 않은 접근"})
        return bool(ok)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, (BASE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if not self._local_only():
            return
        if self.path == "/hero":                # 선택: 이 폴더에 hero.jpg/png/webp를 두면 상단 배경 사진으로 쓴다
            for ext, ct in (("jpg", "image/jpeg"), ("jpeg", "image/jpeg"), ("png", "image/png"), ("webp", "image/webp")):
                f = BASE / f"hero.{ext}"
                if f.is_file() and f.stat().st_size < 8_000_000:
                    return self._send(200, f.read_bytes(), ct)
            return self._send(404, {"error": "no hero"})
        if self.path == "/api/status":
            return self._send(200, auth_status())
        if self.path == "/api/data":
            return self._send(200, load_data())
        if self.path.startswith("/api/job"):
            jid = self.path.split("id=", 1)[-1]
            with JOBS_LOCK:
                j = JOBS.get(jid)
                if not j:
                    return self._send(404, {"error": "작업을 찾을 수 없어요"})
                out = {"status": j["status"], "error": j["error"], "result": j["result"], "elapsed": round((j.get("ended") or time.time()) - j["started"], 1)}
                if j["status"] != "running":
                    del JOBS[jid]            # 결과를 한 번 받아 가면 서버 메모리에서도 지운다
                return self._send(200, out)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._local_only():
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 16_000_000:
                raise ValueError
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "잘못된 요청"})
        if self.path == "/api/run":
            stage = body.get("stage")
            if stage not in prompts.BUILDERS or not isinstance(body.get("data"), dict):
                return self._send(400, {"error": "알 수 없는 요청"})
            return self._send(200, {"job": start_job(stage, body["data"])})
        if self.path == "/api/import":
            try:
                return self._send(200, import_table(body))
            except (ValueError, KeyError, IndexError, zipfile.BadZipFile, ET.ParseError, TypeError, base64.binascii.Error) as e:
                return self._send(200, {"error": str(e) if isinstance(e, ValueError) else "파일을 읽지 못했어요. 엑셀(.xlsx)이나 CSV인지 확인해 주세요"})
        if self.path == "/api/template":
            cols = body.get("cols")
            if not isinstance(cols, list):
                return self._send(400, {"error": "잘못된 요청"})
            return self._send(200, build_template(cols), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if self.path == "/api/data":
            try:
                save_data(body)
            except (ValueError, OSError):
                return self._send(400, {"error": "저장 실패"})
            return self._send(200, {"saved": True})
        if self.path == "/api/login":
            return self._send(200, {"opened": open_login_window()})
        self._send(404, {"error": "not found"})


if __name__ == "__main__":
    if DATA.exists():
        shutil.copyfile(DATA, DATA.with_name("data.backup.json"))   # 시작할 때마다 직전 상태를 백업
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"실험 노트 점검기: http://localhost:{PORT}/   (끄려면 Ctrl+C)")
    print("claude:", find_claude() or "못 찾음")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
