# -*- coding: utf-8 -*-
"""실험실 프로젝트 관리 대시보드 로컬 서버.

실행:  python server.py   → http://localhost:8810/
데이터는 같은 폴더의 data.json 에 저장된다. 이 PC의 브라우저에서 온 요청만 받는다.
"""
import csv
import io
import ipaddress
import json
import os
import re
import socket
import sys
import tempfile
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import xlsx_writer

BASE = Path(__file__).resolve().parent
DATA = BASE / "data.json"
PORT = int(os.environ.get("PORT", "8810"))
# 같은 와이파이의 다른 기기에도 열려면:  python server.py --lan   (보기만 허용하려면 --lan --readonly)
LAN = "--lan" in sys.argv or os.environ.get("LAN") == "1"
READONLY = "--readonly" in sys.argv or os.environ.get("READONLY") == "1"


def host_ok(host):
    name = re.sub(r":\d+$", "", host.strip())
    if name in ("localhost", "127.0.0.1"):
        return True
    if not LAN:
        return False
    try:
        a = ipaddress.ip_address(name)
    except ValueError:
        return False   # 도메인 이름으로 들어온 요청은 받지 않는다
    return a.version == 4 and a.is_private


def client_ok(addr):
    try:
        a = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return a.is_loopback or (LAN and a.version == 4 and a.is_private)
LOCK = threading.Lock()
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def load():
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"projects": []}


def save(obj):
    """임시 파일에 쓴 뒤 교체해서, 저장 중 꺼져도 data.json이 깨지지 않게 한다."""
    fd, tmp = tempfile.mkstemp(dir=BASE, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, DATA)


# ---------- 엑셀 내보내기 ----------
def s(v):
    return "" if v is None else str(v)


DESIGN = "제품 설계"


def col_label(prefix, c):
    name = s(c.get("name")).strip() or "(이름 없음)"
    unit = s(c.get("unit")).strip()
    return f"{prefix}: {name}" + (f" ({unit})" if unit else "")


def exp_parts(rec):
    ex = rec.get("exp") or {}
    return ex.get("x") or [], ex.get("y") or [], ex.get("runs") or []


def structure_summary(rec):
    parts = []
    for l in rec.get("structure") or []:
        bits = [s(l.get("layer")).strip(), s(l.get("material")).strip()]
        if s(l.get("thickness")).strip():
            bits.append(s(l.get("thickness")).strip() + s(l.get("unit")).strip())
        parts.append(" ".join(b for b in bits if b))
    return " / ".join(x for x in parts if x)


def design_sheet(p, rec):
    xs, ys, runs = exp_parts(rec)
    rows = [[(f"{DESIGN}  (시도 {rec.get('attempt', 1)})", "t")], [],
            [("메모", "h")], [(s(rec.get("note")), "wrap")], [],
            [("제품 구조 (바깥 → 안쪽)", "t")],
            [(h, "h") for h in ("순서", "층 이름", "재료·조성", "두께", "단위", "비고")]]
    layers = rec.get("structure") or []
    rows += [[n, l.get("layer"), l.get("material"), l.get("thickness"), l.get("unit"), l.get("note")]
             for n, l in enumerate(layers, 1)] or [["(입력 없음)"]]
    specs = p.get("specs") or []
    rows += [[], [("규격 (평가 기준)", "t")], [(h, "h") for h in ("평가 항목", "규격", "단위")]]
    rows += [[sp.get("item"), sp.get("spec"), sp.get("unit")] for sp in specs] or [["(입력 없음)"]]
    rows += [[], [("실험 데이터 (한 줄 = 실험 1회)", "t")],
             [(h, "h") for h in ["실험번호", "기준(★)", "ML 제외", "날짜", "메모"] + [col_label("X", c) for c in xs] + [col_label("Y", c) for c in ys]]]
    for n, u in enumerate(runs, 1):
        ux, uy = u.get("x") or {}, u.get("y") or {}
        rows.append([n, "★" if u.get("star") else "", ("제외", "fail") if u.get("excl") else "", u.get("date"), u.get("note")]
                    + [ux.get(c.get("id")) for c in xs] + [uy.get(c.get("id")) for c in ys])
    if not runs:
        rows.append(["(입력 없음)"])
    return {"name": f"시도{rec.get('attempt', 1)} {DESIGN}", "widths": [10, 10, 10, 14, 28] + [18] * (len(xs) + len(ys)), "rows": rows}


def compare_sheet(cmp, tol):
    """설계 기준 vs Trial 측정값 비교 (계산은 화면에서 한 결과를 그대로 받아 적는다)"""
    def verdict(v):
        v = s(v)
        return (v, "pass" if v == "일치" else "fail" if v == "불일치" else None)
    rows = [[("설계 vs Trial 성능 비교", "t")], [f"허용 편차 ±{tol}% · 일치=허용 편차 이내, 주의=2배 이내, 불일치=그 이상. 현재 시도의 측정값 기준."], [],
            [(h, "h") for h in ("평가 항목", "단위", "설계 기준값", "기준 종류", "설계 실험 범위", "규격", "최대 편차(%)", "산포 CV(%)", "판정")]]
    for r in cmp.get("summary") or []:
        r = list(r) + [""] * (9 - len(r))
        rows.append(r[:8] + [verdict(r[8])])
    rows += [[], [("측정 이력 (시도·단계별)", "t")],
             [(h, "h") for h in ("평가 항목", "시점", "측정값", "설계 대비 편차(%)", "판정", "구분")]]
    for r in cmp.get("detail") or []:
        r = list(r) + [""] * (6 - len(r))
        res = s(r[4])
        rows.append(r[:4] + [(res, "pass" if res.lower() == "pass" else "fail" if res.lower() == "fail" else None), r[5]])
    return {"name": "성능 비교", "widths": [30, 22, 14, 16, 18, 14, 14, 12, 10], "rows": rows}


def ml_table(data):
    """모든 프로젝트의 '제품 설계' 실험을 한 표(1행=실험 1회)로 합친다. 같은 이름·단위의 열은 한 열로 모인다."""
    xl, yl, recs_out = [], [], []
    for p in data.get("projects", []):
        for rec in p.get("records", []):
            if rec.get("stage") != DESIGN:
                continue
            xs, ys, runs = exp_parts(rec)
            summary = structure_summary(rec)
            for n, u in enumerate(runs, 1):
                if u.get("excl"):      # 데이터 점검에서 '제외'한 실험은 ML 데이터에 넣지 않는다
                    continue
                row = {"프로젝트": s(p.get("name")), "시도": rec.get("attempt", 1), "실험번호": n, "날짜": s(u.get("date")),
                       "제품 구조": summary, "메모": s(u.get("note")), "기준 실험": "1" if u.get("star") else "0"}
                for cols, vals, prefix, lst in ((xs, u.get("x") or {}, "X", xl), (ys, u.get("y") or {}, "Y", yl)):
                    for c in cols:
                        lab = col_label(prefix, c)
                        if lab not in lst:
                            lst.append(lab)
                        v = vals.get(c.get("id"))
                        if v not in (None, ""):
                            row[lab] = v
                recs_out.append(row)
    head = ["프로젝트", "시도", "실험번호", "날짜", "제품 구조", "메모", "기준 실험"] + xl + yl
    return head, [[r.get(h, "") for h in head] for r in recs_out]


LV = {"1": "하", "2": "중", "3": "상"}


def risk_level(k):
    try:
        v = int(k.get("sev") or 0) * int(k.get("prob") or 0)
    except ValueError:
        v = 0
    return "" if not v else "높음" if v >= 6 else "중간" if v >= 3 else "낮음"


RISK_HEAD = ("Risk · 이슈", "분류", "영향", "발생 가능성", "위험도", "상태", "담당", "목표일", "설명 · 원인", "대응 · 해결 방법", "결과 · 확인", "출처")


def risk_row(k, lead=()):
    lvl = risk_level(k)
    st = s(k.get("status"))
    return list(lead) + [k.get("title"), k.get("cat"), LV.get(s(k.get("sev")), ""), LV.get(s(k.get("prob")), ""),
                         (lvl, "fail" if lvl == "높음" else "pass" if lvl == "낮음" else None),
                         (st, "pass" if st == "해결됨" else None), k.get("owner"), k.get("due"),
                         (s(k.get("detail")), "wrap"), (s(k.get("action")), "wrap"), (s(k.get("result")), "wrap"), k.get("from")]


def risk_rows(rec):
    risks = rec.get("risks") or []
    dec = rec.get("decision") or {}
    rows = [[("Trial production 진행 판단", "t")],
            ["회의 결정", dec.get("result") or "미정", "결정일", dec.get("date"), "결정자 · 참석", dec.get("by")],
            ["조건 · 사유", (s(dec.get("cond")), "wrap")], [],
            [("Risk 목록", "t")], [(h, "h") for h in RISK_HEAD]]
    rows += [risk_row(k) for k in risks] or [["(등록된 Risk 없음)"]]
    return rows + [[]]


def risk_history_sheet(p):
    ra = [r for r in p.get("records", []) if isinstance(r, dict) and (r.get("risks") or [])]
    if not ra:
        return None
    order = {"1차 R/A": 1, "2차 R/A": 2, "3차 R/A": 3}
    ra.sort(key=lambda r: (r.get("attempt", 1), order.get(r.get("stage"), 9)))
    rows = [[("이슈 · Risk 이력 (시도·차수별)", "t")], [], [(h, "h") for h in ("시도", "단계") + RISK_HEAD]]
    for r in ra:
        rows += [risk_row(k, (r.get("attempt", 1), r.get("stage"))) for k in r["risks"]]
    return {"name": "Risk 이력", "widths": [7, 12, 30, 12, 8, 12, 9, 10, 9, 12, 36, 36, 36, 30], "rows": rows}


def record_sheet(p, rec):
    if rec.get("stage") == DESIGN:
        return design_sheet(p, rec)
    is_ra = rec.get("risks") is not None or (rec.get("decision") or {}).get("result")
    rows = [[(f"{s(rec.get('stage'))}  (시도 {rec.get('attempt', 1)})", "t")], [],
            *(risk_rows(rec) if is_ra else []),
            [("메모", "h")], [(s(rec.get("note")), "wrap")], [],
            [("Recipe", "t")],
            [(h, "h") for h in ("항목", "값", "단위", "비고")]]
    recipe = rec.get("recipe") or []
    rows += [[r.get("item"), r.get("value"), r.get("unit"), r.get("note")] for r in recipe] or [["(입력 없음)"]]
    rows += [[], [("특성 평가 결과", "t")], [(h, "h") for h in ("평가 항목", "규격(기준)", "측정값", "단위", "판정", "비고")]]
    evals = rec.get("evals") or []
    for e in evals:
        res = s(e.get("result"))
        rows.append([e.get("item"), e.get("spec"), e.get("value"), e.get("unit"),
                     (res, "pass" if res.lower() == "pass" else "fail" if res.lower() == "fail" else None), e.get("note")])
    if not evals:
        rows.append(["(입력 없음)"])
    return {"name": f"시도{rec.get('attempt', 1)} {s(rec.get('stage'))}", "widths": [30, 18, 14, 12, 10, 36, 12, 12, 36, 36, 36, 30], "rows": rows}


def summary_sheet(p, recs, scope):
    rows = [[("실험실 프로젝트 관리 대시보드 내보내기", "t")], [],
            ["프로젝트", p.get("name")], ["범위", scope], ["내보낸 날짜", date.today().isoformat()],
            ["현재 단계", p.get("stage_name")], ["연속 통과", p.get("passed_text")], ["현재 시도", p.get("attempt")], [],
            [("재개 메모", "h")], [(s(p.get("note")), "wrap")]]
    if scope == "전체":
        rows += [[], [("진행 기록", "h")]] + [[l.get("date"), l.get("text")] for l in p.get("log", [])]
    return {"name": "요약", "widths": [16, 70], "rows": rows}


def export(body):
    p = body["project"]
    stage, attempt = body.get("stage"), body.get("attempt")
    recs = [r for r in p.get("records", []) if isinstance(r, dict)]
    if stage:
        recs = [r for r in recs if r.get("stage") == stage and r.get("attempt") == attempt]
        scope = f"{stage} (시도 {attempt})"
    else:
        # 화면에서 열어만 보고 비워 둔 단계는 빈 시트로 내보내지 않는다
        recs = [r for r in recs if s(r.get("note")).strip() or r.get("recipe") or r.get("evals")
                or r.get("structure") or (r.get("exp") or {}).get("runs") or r.get("risks")
                or (r.get("decision") or {}).get("result")]
        scope = "전체"
    cmp = body.get("compare") if isinstance(body.get("compare"), dict) else None
    if body.get("compare_only") and cmp:
        recs, scope = [], "설계 vs Trial 성능 비교"
    sheets = [summary_sheet(p, recs, scope)]
    if cmp and not stage:
        sheets.append(compare_sheet(cmp, cmp.get("tol", 5)))
    if not stage and not body.get("compare_only"):
        rh = risk_history_sheet(p)
        if rh:
            sheets.append(rh)
    sheets += [record_sheet(p, r) for r in recs]
    base = re.sub(r'[\\/:*?"<>|]+', "_", s(p.get("name")).strip() or "프로젝트")[:60]
    tail = f"{stage}_시도{attempt}" if stage else "성능비교" if body.get("compare_only") else "전체"
    name = re.sub(r'[\\/:*?"<>|]+', "_", f"{base}_{tail}") + ".xlsx"
    return xlsx_writer.build(sheets), name


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _local_only(self):
        """접근 제어. 기본은 이 PC(localhost)에서만. --lan 이면 같은 내부망(사설 IP)에서 IP 주소로 접속한 요청까지 허용한다.
        Host가 IP·localhost가 아니면(다른 사이트가 이 서버를 몰래 부르는 경우 등) 거부하고,
        Origin이 있으면 접속한 주소와 같아야 한다."""
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin", "")
        ok = host_ok(host) and client_ok(self.client_address[0]) and (not origin or origin == "http://" + host)
        if not ok:
            self._send(403, {"error": "허용되지 않은 접근"})
        return bool(ok)

    def _body(self):
        n = int(self.headers.get("Content-Length", "0"))
        if n > 10_000_000:
            raise ValueError("too large")
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self):
        if self.path.startswith("/api/ml"):
            if not self._local_only():
                return
            with LOCK:
                head, rows = ml_table(load())
            if "fmt=csv" in self.path:
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerow(head)
                w.writerows(rows)
                data, ctype, ext = ("﻿" + buf.getvalue()).encode("utf-8"), "text/csv; charset=utf-8", "csv"
            else:
                sheet = {"name": "실험 데이터", "widths": [30, 6, 10, 12, 50, 24, 10] + [18] * (len(head) - 7),
                         "rows": [[(h, "h") for h in head]] + rows}
                data, ctype, ext = xlsx_writer.build([sheet]), XLSX, "xlsx"
            name = quote(f"ML용_실험데이터_{date.today().isoformat()}.{ext}")
            return self._send(200, data, ctype, {"Content-Disposition": f"attachment; filename*=UTF-8''{name}"})
        if self.path in ("/", "/index.html"):
            return self._send(200, (BASE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/data":
            if self._local_only():
                with LOCK:
                    self._send(200, load())
            return
        self._send(404, {"error": "not found"})

    def do_PUT(self):
        if self.path != "/api/data" or not self._local_only():
            return
        if READONLY:
            return self._send(403, {"error": "읽기 전용으로 열려 있어서 저장할 수 없어요"})
        try:
            obj = self._body()
            if not isinstance(obj.get("projects"), list):
                raise ValueError
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "잘못된 데이터"})
        with LOCK:
            save(obj)
        self._send(200, {"ok": True})

    def do_POST(self):
        if not self._local_only():
            return
        try:
            body = self._body()
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "잘못된 요청"})
        if self.path == "/api/export":
            try:
                data, name = export(body)
            except (KeyError, TypeError, AttributeError):
                return self._send(400, {"error": "내보낼 데이터가 올바르지 않아요"})
            return self._send(200, data, XLSX, {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})
        self._send(404, {"error": "not found"})


def lan_addresses():
    """이 PC의 내부망 IPv4 주소들 (같은 와이파이의 다른 기기가 접속할 때 쓰는 주소)"""
    found = []
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            found.append(ip)
    except OSError:
        pass
    try:   # 실제로 패킷을 보내지 않고, 기본 경로로 쓰이는 내 주소만 알아낸다
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        found.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    out = []
    for ip in found:
        try:
            a = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if a.version == 4 and a.is_private and not a.is_loopback and not a.is_link_local and ip not in out:
            out.append(ip)
    return out


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0" if LAN else "127.0.0.1", PORT), Handler)
    print(f"실험실 프로젝트 관리 대시보드: http://localhost:{PORT}/   (끄려면 Ctrl+C)")
    if LAN:
        print("같은 와이파이의 다른 기기에서는 아래 주소로 접속해요:")
        for ip in lan_addresses():
            print(f"    http://{ip}:{PORT}/")
        print("※ 이 주소를 아는 같은 네트워크의 누구나 열 수 있어요." + (" (읽기 전용: 저장은 막혀 있어요)" if READONLY else " 입력·수정·삭제도 가능하니 믿을 수 있는 사람과만 쓰세요."))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
