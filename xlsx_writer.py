# -*- coding: utf-8 -*-
"""외부 라이브러리 없이 .xlsx 를 만드는 최소 작성기 (zipfile + XML).

sheet = {"name": str, "widths": [int,...], "rows": [[cell,...], ...]}
cell  = 값  또는  (값, 스타일)   스타일: "h"=표 머리글, "t"=제목, "pass", "fail", "wrap"
"""
import io
import re
import zipfile
from xml.sax.saxutils import escape

STYLE = {None: 0, "h": 1, "t": 2, "pass": 3, "fail": 4, "wrap": 5}
_NUM = re.compile(r"^-?(0|[1-9]\d*)(\.\d+)?$")
_BAD_SHEET = re.compile(r"[\[\]:*?/\\]")


def col_name(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def clean_xml(text):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(text))


def cell_xml(ref, cell):
    value, style = cell if isinstance(cell, tuple) else (cell, None)
    s = STYLE.get(style, 0)
    if value is None or value == "":
        return f'<c r="{ref}" s="{s}"/>' if s else ""
    text = clean_xml(value).strip() if isinstance(value, str) else value
    if isinstance(text, (int, float)) or (isinstance(text, str) and _NUM.match(text) and len(text) < 16):
        return f'<c r="{ref}" s="{s}"><v>{text}</v></c>'
    return f'<c r="{ref}" s="{s}" t="inlineStr"><is><t xml:space="preserve">{escape(str(text))}</t></is></c>'


def sheet_xml(sheet):
    cols = "".join(f'<col min="{i+1}" max="{i+1}" width="{w}" customWidth="1"/>' for i, w in enumerate(sheet.get("widths", [])))
    rows = []
    for r, row in enumerate(sheet["rows"], 1):
        cells = "".join(cell_xml(f"{col_name(c)}{r}", cell) for c, cell in enumerate(row))
        rows.append(f'<row r="{r}">{cells}</row>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<cols>{cols}</cols><sheetData>{"".join(rows)}</sheetData></worksheet>')


STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="5">
<font><sz val="11"/><name val="Malgun Gothic"/></font>
<font><b/><sz val="11"/><name val="Malgun Gothic"/></font>
<font><b/><sz val="14"/><name val="Malgun Gothic"/></font>
<font><b/><sz val="11"/><color rgb="FF1A7F4B"/><name val="Malgun Gothic"/></font>
<font><b/><sz val="11"/><color rgb="FFC0392B"/><name val="Malgun Gothic"/></font>
</fonts>
<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FFE8EEFC"/><bgColor indexed="64"/></patternFill></fill></fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="6">
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
<xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1"/>
<xf numFmtId="0" fontId="3" fillId="0" borderId="0" xfId="0" applyFont="1"/>
<xf numFmtId="0" fontId="4" fillId="0" borderId="0" xfId="0" applyFont="1"/>
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf>
</cellXfs>
</styleSheet>"""


def safe_sheet_name(name, used):
    base = _BAD_SHEET.sub("-", clean_xml(name)).strip("'") or "Sheet"
    base = base[:31]
    out, n = base, 2
    while out in used:
        suffix = f" ({n})"
        out = base[:31 - len(suffix)] + suffix
        n += 1
    used.add(out)
    return out


def build(sheets):
    used, names = set(), []
    for s in sheets:
        names.append(safe_sheet_name(s["name"], used))
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
          '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
          + "".join(f'<Override PartName="/xl/worksheets/sheet{i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(len(sheets)))
          + '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    wb = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
          + "".join(f'<sheet name="{escape(n, {chr(34): "&quot;"})}" sheetId="{i+1}" r:id="rId{i+1}"/>' for i, n in enumerate(names))
          + '</sheets></workbook>')
    wbrels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
              + "".join(f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i+1}.xml"/>' for i in range(len(sheets)))
              + f'<Relationship Id="rId{len(sheets)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("xl/workbook.xml", wb)
        z.writestr("xl/_rels/workbook.xml.rels", wbrels)
        z.writestr("xl/styles.xml", STYLES)
        for i, s in enumerate(sheets):
            z.writestr(f"xl/worksheets/sheet{i+1}.xml", sheet_xml(s))
    return buf.getvalue()
