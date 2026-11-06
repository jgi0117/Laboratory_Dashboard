# -*- coding: utf-8 -*-
"""python -m unittest -v  (저장소 루트에서)"""
import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab import dataset, importer, migrate, ml           # noqa: E402
from lab.store import Store, StoreError                 # noqa: E402

SAMPLE = ROOT / "data.json"


def fresh_store(tmp):
    shutil.copy(SAMPLE, tmp / "data.json")
    return Store(tmp / "data.json")


class TestSpec(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(dataset.parse_spec("≥70")["min"], 70)
        self.assertEqual(dataset.parse_spec(">= 70")["min"], 70)
        self.assertEqual(dataset.parse_spec("≤1.5")["max"], 1.5)
        self.assertEqual(dataset.parse_spec("70~80")["min"], 70)
        self.assertEqual(dataset.parse_spec("100±5")["max"], 105)
        self.assertEqual(dataset.parse_spec("70")["target"], 70)
        self.assertIsNone(dataset.parse_spec("abc"))

    def test_judge(self):
        self.assertEqual(dataset.judge("≥70", 71), "Pass")
        self.assertEqual(dataset.judge("≥70", "69.9"), "Fail")
        self.assertEqual(dataset.judge("≤1.5", 1.5), "Pass")
        self.assertEqual(dataset.judge("70~80", 85), "Fail")
        self.assertEqual(dataset.judge("70", 70), "")
        self.assertEqual(dataset.judge("≥70", None), "")


class TestMigrate(unittest.TestCase):
    def test_v1_to_v2(self):
        old = json.loads(SAMPLE.read_text(encoding="utf-8"))
        if old.get("version") == 2:
            bk = ROOT / "data.v1.backup.json"
            if not bk.exists():
                self.skipTest("v1 데이터 없음")
            old = json.loads(bk.read_text(encoding="utf-8"))
        new = migrate.v1_to_v2(old)
        self.assertEqual(new["version"], 2)
        self.assertEqual(len(new["projects"]), 10)
        self.assertEqual(sum(1 for e in new["experiments"] if e["source"] == "migrated:설계"), 106)
        self.assertEqual({v["kind"] for v in new["variables"]}, {"composition", "process", "property"})


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.s = fresh_store(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_loaded(self):
        self.assertEqual(self.s.data["version"], 2)
        self.assertGreater(len(self.s.experiments()), 100)

    def test_project_crud(self):
        p = self.s.add_project({"name": "테스트", "product": "필름"})
        self.assertTrue(p["id"].startswith("p"))
        self.s.update_project(p["id"], {"status": "보류", "specs": [{"var": "v0009", "spec": "≥60"}]})
        self.assertEqual(self.s.project(p["id"])["status"], "보류")
        with self.assertRaises(StoreError):
            self.s.update_project(p["id"], {"status": "이상한값"})
        with self.assertRaises(StoreError):
            self.s.update_project(p["id"], {"name": ""})
        e = self.s.add_experiments([{"project": p["id"], "values": {"v0001": "12", "v0009": "71.2", "v0002": ""}}])[0]
        self.assertEqual(e["values"], {"v0001": 12.0, "v0009": 71.2})
        self.s.update_experiment(e["id"], {"set": {"v0001": None, "v0013": "1.1"}})
        self.assertEqual(self.s.experiment(e["id"])["values"], {"v0009": 71.2, "v0013": 1.1})
        self.s.delete_project(p["id"])
        with self.assertRaises(StoreError):
            self.s.experiment(e["id"])

    def test_variable_merge_and_delete(self):
        v = self.s.add_variable("나노입자 함량", "wt%", "composition")
        self.assertEqual(v["id"], "v0001")                   # 같은 이름·단위면 기존 변수
        nv = self.s.add_variable("NP 함량", "wt%", "composition")
        self.s.add_experiments([{"project": "p0001", "values": {nv["id"]: 9}}])
        with self.assertRaises(StoreError):
            self.s.delete_variable(nv["id"])
        d = self.s.merge_variables(nv["id"], "v0001")
        self.assertIn("NP 함량", d["aliases"])
        self.assertIsNone(next((x for x in self.s.variables() if x["id"] == nv["id"]), None))
        self.assertEqual(self.s.experiments()[-1]["values"]["v0001"], 9.0)
        self.assertIs(self.s.find_variable("np 함량"), d)

    def test_save_roundtrip(self):
        self.s.add_project({"name": "저장"})
        self.s.save()
        s2 = Store(self.tmp / "data.json")
        self.assertEqual(s2.projects()[-1]["name"], "저장")


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.s = fresh_store(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_table_and_export(self):
        t = dataset.table(self.s)
        self.assertEqual(len(t["rows"]), len(self.s.experiments()))
        head, rows = dataset.export_rows(self.s, ["p0001"])
        self.assertTrue(head[0] == "프로젝트" and len(rows) == len(self.s.experiments("p0001")))
        self.assertTrue(dataset.to_csv(head, rows).startswith("﻿".encode("utf-8")))
        self.assertEqual(dataset.to_xlsx(head, rows)[:2], b"PK")

    def test_summary_and_checks(self):
        sm = dataset.project_summary(self.s)
        self.assertEqual(sm["p0001"]["n"], len(self.s.experiments("p0001")))
        self.assertTrue(any(c["type"] == "outlier" for c in dataset.checks(self.s)))


class TestImporter(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.s = fresh_store(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_csv_mapping_and_commit(self):
        csv = ("실험,날짜,나노입자 함량 (wt%),바인더 함량 비율(wt%),안료 종류,Haze (%),비고\n"
               "T-1,2026-03-01,11,33.0,카본,1.2,ok\n"
               "T-2,2026/03/02,약 12,34,카본,1.4,\n"
               ",,,,,,\n").encode("utf-8")
        pv = importer.preview("test.csv", csv, self.s)
        acts = {m["header"]: m["action"] for m in pv["mapping"]}
        self.assertEqual(acts["실험"], "label")
        self.assertEqual(acts["날짜"], "date")
        self.assertEqual(acts["비고"], "note")
        self.assertEqual(acts["나노입자 함량 (wt%)"], "var")
        self.assertEqual(acts["바인더 함량 비율(wt%)"], "var")   # 비슷한 이름은 기존 변수로 제안
        self.assertEqual(acts["안료 종류"], "new")
        self.assertEqual(pv["n"], 2)
        r = importer.commit(self.s, "p0001", pv["headers"], pv["mapping"], pv["rows"], source="import:test.csv")
        self.assertEqual(r["added"], 2)
        self.assertEqual(r["new_variables"], 1)
        e = self.s.experiments("p0001")[-2:]
        self.assertEqual(e[0]["label"], "T-1")
        self.assertEqual(e[1]["date"], "2026-03-02")
        self.assertEqual(e[1]["values"]["v0001"], 12.0)         # '약 12' → 12
        self.assertEqual(e[0]["values"]["v0002"], 33.0)
        pig = self.s.find_variable("안료 종류")
        self.assertEqual(e[0]["values"][pig["id"]], "카본")
        self.assertIn("바인더 함량 비율", self.s.variable("v0002")["aliases"])   # 열 이름을 별칭으로 기억

    def test_cp949_and_tab(self):
        txt = "실험\tHaze (%)\nA\t1.1\n".encode("cp949")
        hd, rows = importer.read_csv_bytes(txt)
        self.assertEqual(hd, ["실험", "Haze (%)"])
        self.assertEqual(rows[0][1], "1.1")

    def test_xlsx_roundtrip(self):
        import xlsx_writer
        data = xlsx_writer.build([{"name": "s", "widths": [10, 10], "rows": [[("실험", "h"), ("Haze (%)", "h")], ["A", 1.25], ["B", "약 2"]]}])
        hd, rows = importer.read_xlsx_bytes(data)
        self.assertEqual(hd, ["실험", "Haze (%)"])
        self.assertEqual(rows[0], ["A", "1.25"])
        self.assertEqual(rows[1][1], "약 2")


@unittest.skipUnless(ml.AVAILABLE, "numpy/scikit-learn 없음")
class TestML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.s = fresh_store(cls.tmp)
        cls.eng = ml.Engine(cls.s, cls.tmp / "models.pkl")
        cls.report = cls.eng.train()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_train_quality(self):
        ok = {t["name"]: t for t in self.report["targets"] if t["ok"]}
        self.assertGreaterEqual(len(ok), 5)
        for name in ("가시광선 투과율(VLT)", "IR 차단율 (940nm)", "Haze"):
            self.assertGreaterEqual(ok[name]["r2"], 0.7, name)
        self.assertNotIn("v0003", self.eng.features)            # 상수 열은 제외
        self.assertLess(self.eng.bounds["v0006"][1], 200)        # 이상치(1050)는 범위에서 제외

    def test_predict(self):
        r = self.eng.predict({"v0001": 12, "v0002": 32}, specs=self.s.project("p0001")["specs"], project="p0001")
        t = {x["name"]: x for x in r["targets"] if x["ok"]}
        self.assertTrue(t["Haze"]["lo"] <= t["Haze"]["pred"] <= t["Haze"]["hi"])
        self.assertIn(t["Haze"]["result"], ("Pass", "Fail"))
        self.assertIn("v0004", r["missing"])
        self.assertEqual(r["project_warning"], "")
        self.assertEqual(len(r["nearest"]), 3)
        r2 = self.eng.predict({"v0001": 40}, project="p_new")
        self.assertIn("v0001", r2["out_of_range"])
        self.assertTrue(r2["project_warning"])

    def test_inverse(self):
        r = self.eng.inverse(self.s.project("p0001")["specs"], fixed={"v0006": 105}, project="p0001")
        self.assertTrue(r["feasible"])
        self.assertEqual(len(r["candidates"]), 5)
        for c in r["candidates"]:
            self.assertEqual(c["values"]["v0006"], 105.0)
            for f, v in c["values"].items():
                lo, hi = self.eng.bounds[f]
                self.assertTrue(lo - 1e-6 <= v <= hi + 1e-6, f)
        self.assertTrue(r["candidates"][0]["all_pass"])
        hard = self.eng.inverse([{"var": "v0009", "spec": "≥95"}], project="p0005")
        self.assertFalse(hard["feasible"])
        self.assertTrue(hard["message"])

    def test_sensitivity_and_cache(self):
        s = self.eng.sensitivity("v0013", "v0004", {"v0001": 12}, project="p0001")
        self.assertEqual(len(s["x"]), 25)
        e2 = ml.Engine(self.s, self.tmp / "models.pkl")
        self.assertTrue(e2.status()["trained"])
        self.assertFalse(e2.status()["stale"])
        self.s.add_experiments([{"project": "p0001", "values": {"v0001": 10}}])
        self.assertTrue(e2.status()["stale"])


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        shutil.copy(SAMPLE, cls.tmp / "data.json")
        import server
        server.STORE = Store(cls.tmp / "data.json")
        server.ENGINE = ml.Engine(server.STORE, None)
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def call(self, method, path, body=None, host=None):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json", **({"Host": host} if host else {})})
        try:
            with urllib.request.urlopen(req) as r:
                raw = r.read()
                return r.status, json.loads(raw.decode("utf-8")) if "json" in r.headers.get("Content-Type", "") else raw
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def test_api(self):
        st, d = self.call("GET", "/api/data")
        self.assertEqual(st, 200)
        self.assertEqual(d["version"], 2)
        self.assertIn("ml", d)
        st, r = self.call("POST", "/api/projects", {"name": "API 프로젝트"})
        pid = r["project"]["id"]
        st, r = self.call("POST", "/api/experiments", [{"project": pid, "values": {"v0001": 10}}])
        eid = r["experiments"][0]["id"]
        st, r = self.call("PATCH", f"/api/experiments/{eid}", {"set": {"v0009": "70"}})
        self.assertEqual(r["experiment"]["values"]["v0009"], 70.0)
        st, r = self.call("PATCH", "/api/projects/nope", {"name": "x"})
        self.assertEqual(st, 400)
        st, csv = self.call("GET", f"/api/dataset?fmt=csv&projects={pid}")
        self.assertIn("API 프로젝트".encode("utf-8"), csv)
        st, r = self.call("DELETE", f"/api/projects/{pid}")
        self.assertEqual(st, 200)
        st, r = self.call("GET", "/api/data", host="evil.example.com")
        self.assertEqual(st, 403)
        st, r = self.call("POST", "/api/ml/predict", {"values": {}})
        self.assertEqual(st, 400)        # 학습 전


if __name__ == "__main__":
    unittest.main()
