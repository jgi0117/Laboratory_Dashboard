# -*- coding: utf-8 -*-
"""실험 데이터 허브 + ML 예측 로컬 서버 (제품개발팀용).

실행:  python server.py            → http://localhost:8810/
       python server.py --lan      → 같은 와이파이의 다른 기기도 접속 (사설 IP만)
       python server.py --readonly → 보기만 허용

데이터는 같은 폴더의 data.json, 학습한 모델은 models.pkl 에 저장된다. 외부로 나가는 것은 없다.
"""
import base64
import copy
import ipaddress
import json
import os
import re
import socket
import sys
import threading
import time
import traceback
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from lab import dataset, importer, ml          # noqa: E402
from lab.store import Store, StoreError        # noqa: E402

DATA = BASE / "data.json"
MODELS = BASE / "models.pkl"
PORT = int(os.environ.get("PORT", "8810"))
LAN = "--lan" in sys.argv or os.environ.get("LAN") == "1"
READONLY = "--readonly" in sys.argv or os.environ.get("READONLY") == "1"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

STORE = Store(DATA)
ENGINE = ml.Engine(STORE, MODELS)
ML_LOCK = threading.Lock()
TRAIN = {"running": False, "error": "", "seconds": None}
TRAIN_LOCK = threading.Lock()


def train_job():
    """학습은 데이터 스냅샷으로 백그라운드에서 돌리고, 끝나면 모델만 바꿔 끼운다 (학습 중에도 조회·입력 가능)."""
    t0 = time.time()
    try:
        with STORE.lock:
            snap = Store.from_data(copy.deepcopy(STORE.data))
        eng = ml.Engine(snap, None)
        eng.train()
        with ML_LOCK:
            for k in ml.CACHE_KEYS:
                setattr(ENGINE, k, getattr(eng, k))
            ENGINE._save_cache()
        TRAIN["error"] = ""
    except Exception as e:          # 학습 실패는 상태로 알린다
        TRAIN["error"] = str(e)
    finally:
        TRAIN["seconds"] = round(time.time() - t0, 1)
        TRAIN["running"] = False


def start_training():
    with TRAIN_LOCK:
        if TRAIN["running"]:
            return {"started": False, "running": True}
        TRAIN["running"], TRAIN["error"] = True, ""
        threading.Thread(target=train_job, daemon=True).start()
        return {"started": True, "running": True}


def ml_status():
    with ML_LOCK:
        st = ENGINE.status()
    st.update(training=TRAIN["running"], train_error=TRAIN["error"], train_seconds=TRAIN["seconds"])
    return st


# ---------- 접근 제어 (v1과 같음) ----------
def host_ok(host):
    name = re.sub(r":\d+$", "", host.strip())
    if name in ("localhost", "127.0.0.1"):
        return True
    if not LAN:
        return False
    try:
        a = ipaddress.ip_address(name)
    except ValueError:
        return False
    return a.version == 4 and a.is_private


def client_ok(addr):
    try:
        a = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return a.is_loopback or (LAN and a.version == 4 and a.is_private)


def _jsonable(o):
    """numpy 숫자/불리언 등 json이 모르는 값을 파이썬 값으로."""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    # ---------- 응답 ----------
    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, default=_jsonable).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _file(self, data, ctype, name):
        self._send(200, data, ctype, {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})

    def _allowed(self):
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin", "")
        ok = host_ok(host) and client_ok(self.client_address[0]) and (not origin or origin == "http://" + host)
        if not ok:
            self._send(403, {"error": "허용되지 않은 접근"})
        return bool(ok)

    def _body(self):
        n = int(self.headers.get("Content-Length", "0"))
        if n > 30_000_000:
            raise ValueError("too large")
        raw = self.rfile.read(n)
        return json.loads(raw.decode("utf-8")) if n else {}

    def _route(self):
        u = urlparse(self.path)
        return u.path.rstrip("/") or "/", {k: v[0] for k, v in parse_qs(u.query).items()}

    # ---------- GET ----------
    def do_GET(self):
        try:
            self._get()
        except Exception as e:              # 어떤 오류든 연결을 끊지 않고 500으로 알린다
            traceback.print_exc()
            self._send(500, {"error": f"서버 오류: {e.__class__.__name__}: {e}"})

    def _get(self):
        path, q = self._route()
        if path in ("/", "/index.html"):
            return self._send(200, (BASE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path.startswith("/docs/"):
            f = (BASE / unquote(path).lstrip("/")).resolve()
            types = {".html": "text/html; charset=utf-8", ".md": "text/plain; charset=utf-8", ".png": "image/png",
                     ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}
            if f.is_file() and BASE in f.parents and f.suffix.lower() in types:
                ct = types[f.suffix.lower()]
                return self._send(200, f.read_bytes(), ct)
            return self._send(404, {"error": "not found"})
        if not path.startswith("/api/") or not self._allowed():
            return self._send(404, {"error": "not found"}) if not path.startswith("/api/") else None
        try:
            if path == "/api/data":
                with STORE.lock:
                    d = STORE.data
                    body = {"version": 2, "variables": d["variables"], "projects": d["projects"], "experiments": d["experiments"],
                            "summary": dataset.project_summary(STORE), "stats": dataset.var_stats(STORE),
                            "checks": dataset.checks(STORE), "readonly": READONLY, "ml": ml_status()}
                return self._send(200, body)
            if path == "/api/dataset":
                with STORE.lock:
                    pids = [p for p in (q.get("projects") or "").split(",") if p] or None
                    head, rows = dataset.export_rows(STORE, pids)
                name = f"실험데이터_{date.today().isoformat()}"
                if q.get("fmt") == "xlsx":
                    return self._file(dataset.to_xlsx(head, rows), XLSX, name + ".xlsx")
                return self._file(dataset.to_csv(head, rows), "text/csv; charset=utf-8", name + ".csv")
            if path == "/api/ml/status":
                return self._send(200, ml_status())
        except StoreError as e:
            return self._send(400, {"error": str(e)})
        self._send(404, {"error": "not found"})

    # ---------- 쓰기 ----------
    def _write_guard(self):
        if READONLY:
            self._send(403, {"error": "읽기 전용으로 열려 있어서 저장할 수 없어요"})
            return False
        return True

    def do_POST(self):
        self._mutate("POST")

    def do_PATCH(self):
        self._mutate("PATCH")

    def do_DELETE(self):
        self._mutate("DELETE")

    def _mutate(self, method):
        try:
            self._mutate_inner(method)
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": f"서버 오류: {e.__class__.__name__}: {e}"})

    def _mutate_inner(self, method):
        path, q = self._route()
        if not path.startswith("/api/") or not self._allowed():
            return
        try:
            body = self._body()
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "잘못된 요청"})
        try:
            # ----- ML (읽기 전용 모드에서도 예측은 가능, 학습은 캐시를 쓰므로 막는다) -----
            if path == "/api/ml/train":
                if not self._write_guard():
                    return
                return self._send(200, start_training())
            if path == "/api/ml/explore":
                with ML_LOCK:
                    return self._send(200, ENGINE.explore(body.get("specs"), body.get("fixed"), body.get("project"), n=int(body.get("n") or 3)))
            if path == "/api/ml/pareto":
                with ML_LOCK:
                    return self._send(200, ENGINE.pareto(body.get("objectives") or [], body.get("specs"), body.get("fixed"), body.get("project")))
            if path == "/api/ml/predict":
                with ML_LOCK:
                    return self._send(200, ENGINE.predict(body.get("values") or {}, body.get("specs"), body.get("project")))
            if path == "/api/ml/inverse":
                with ML_LOCK:
                    return self._send(200, ENGINE.inverse(body.get("specs") or [], body.get("fixed"), body.get("project"),
                                                          n=int(body.get("n") or 5)))
            if path == "/api/ml/sensitivity":
                with ML_LOCK:
                    return self._send(200, ENGINE.sensitivity(body.get("target"), body.get("var"), body.get("base"), body.get("project")))
            if path == "/api/ml/similar":
                with ML_LOCK:
                    return self._send(200, {"nearest": ENGINE.similar(body.get("values") or {}, body.get("project"), int(body.get("k") or 5))})
            # ----- 가져오기 미리보기 (저장하지 않음) -----
            if path == "/api/import/preview":
                raw = base64.b64decode(body.get("data") or "") if body.get("data") else (body.get("text") or "").encode("utf-8")
                with STORE.lock:
                    return self._send(200, importer.preview(body.get("name") or "paste.csv", raw, STORE, limit=int(body.get("limit") or 15)))
            if not self._write_guard():
                return
            with STORE.lock:
                res = self._write(method, path, body)
                STORE.save()
            return self._send(200, res)
        except (StoreError, ml.NotTrained, RuntimeError) as e:
            return self._send(400, {"error": str(e)})
        except (KeyError, TypeError, AttributeError) as e:
            traceback.print_exc()
            return self._send(400, {"error": f"요청 형식이 올바르지 않아요 ({e.__class__.__name__}: {e})"})

    def _write(self, method, path, body):
        parts = path.split("/")[2:]          # ['projects', 'p0001']
        coll = parts[0] if parts else ""
        oid = parts[1] if len(parts) > 1 else None
        if coll == "projects":
            if method == "POST":
                return {"project": STORE.add_project(body)}
            if method == "PATCH" and oid:
                return {"project": STORE.update_project(oid, body)}
            if method == "DELETE" and oid:
                STORE.delete_project(oid)
                return {"ok": True}
        if coll == "experiments":
            if method == "POST":
                items = body if isinstance(body, list) else body.get("items") or [body]
                return {"experiments": STORE.add_experiments(items)}
            if method == "PATCH" and oid:
                return {"experiment": STORE.update_experiment(oid, body)}
            if method == "DELETE" and oid:
                STORE.delete_experiment(oid)
                return {"ok": True}
        if coll == "variables":
            if method == "POST":
                if body.get("merge"):
                    return {"variable": STORE.merge_variables(body["merge"]["src"], body["merge"]["dst"])}
                return {"variable": STORE.add_variable(body.get("name"), body.get("unit", ""), body.get("kind", "process"),
                                                       body.get("aliases"), body.get("desc", ""))}
            if method == "PATCH" and oid:
                return {"variable": STORE.update_variable(oid, body)}
            if method == "DELETE" and oid:
                STORE.delete_variable(oid, force=bool(body.get("force")))
                return {"ok": True}
        if coll == "import" and oid == "commit" and method == "POST":
            return importer.commit(STORE, body["project"], body["headers"], body["mapping"], body["rows"],
                                   source="import:" + str(body.get("name") or "file"))
        raise StoreError("없는 경로")


def lan_addresses():
    found = []
    try:
        found += socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        pass
    try:
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
    s = STORE.summary()
    print(f"실험 데이터 허브: http://localhost:{PORT}/   (끄려면 Ctrl+C)")
    print(f"데이터: 프로젝트 {s['projects']}개 · 실험 {s['experiments']}건 · 변수 {s['variables']}개"
          + ("" if ENGINE.available else "   ※ ML 꺼짐: pip install numpy scikit-learn"))
    if LAN:
        print("같은 와이파이의 다른 기기에서는 아래 주소로 접속해요:")
        for ip in lan_addresses():
            print(f"    http://{ip}:{PORT}/")
        print("※ 이 주소를 아는 같은 네트워크의 누구나 열 수 있어요." + (" (읽기 전용)" if READONLY else " 입력·수정·삭제도 가능하니 믿을 수 있는 사람과만 쓰세요."))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
