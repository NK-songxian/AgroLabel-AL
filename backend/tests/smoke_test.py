#!/usr/bin/env python
"""后端冒烟测试：对着跑起来的服务逐条验契约。

用法（仓库根目录）：
    # 先起服务
    .\\.venv\\Scripts\\python.exe -m uvicorn backend.app:app --port 8000
    # 再跑本脚本
    .\\.venv\\Scripts\\python.exe backend/tests/smoke_test.py

只依赖标准库，不需要 pytest。每步打印 PASS/FAIL，末尾汇总退出码。
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"

passed = failed = 0


def check(name: str, cond: bool, detail: str = "") -> bool:
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f"  [{detail}]" if detail else ""))
    return cond


def http(method: str, path: str, body: dict | None = None,
         headers: dict | None = None):
    """返回 (status, headers, raw_bytes)。headers 的键统一小写（uvicorn 会小写化）。

    4xx/5xx 不当异常抛，交给断言判断。
    """
    url = BASE + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, _lower(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, _lower(e.headers), e.read()


def _lower(headers) -> dict:
    return {k.lower(): v for k, v in headers.items()}


def get_json(path: str):
    st, _, raw = http("GET", path)
    return st, json.loads(raw.decode("utf-8"))


def main() -> int:
    print("=" * 64)
    print("AgroLabel-AL 后端冒烟测试")
    print("=" * 64)

    # ---------------------------------------------------- health
    print("\n[1] GET /api/health")
    st, h = get_json("/api/health")
    check("200", st == 200, str(st))
    check("status=ok", h.get("status") == "ok")
    check("classes=34", h.get("classes") == 34, str(h.get("classes")))
    ds = h.get("dataset", {})
    check("dataset.mode 合法", ds.get("mode") in ("manifest", "demo"), ds.get("mode"))
    check("dataset.total > 0", (ds.get("total") or 0) > 0, str(ds.get("total")))
    mi = h.get("model_info", {})
    check("model_info.device 存在", bool(mi.get("device")), mi.get("device"))
    check("model_version 带 sha256", "@sha256:" in (mi.get("model_version") or ""),
          mi.get("model_version", "")[:48])

    # ---------------------------------------------------- classes
    print("\n[2] GET /api/classes")
    st, c = get_json("/api/classes")
    check("200", st == 200, str(st))
    check("nc=34", c.get("nc") == 34, str(c.get("nc")))
    cls = c.get("classes", [])
    check("classes 长度=34", len(cls) == 34, str(len(cls)))
    check("class_id 连续 0..33", [x["class_id"] for x in cls] == list(range(34)))
    check("首类是 Apple", cls[0]["class_name"] == "Apple", cls[0]["class_name"])

    # ---------------------------------------------------- images
    print("\n[3] GET /api/images")
    st, page = get_json("/api/images?limit=3")
    check("200", st == 200, str(st))
    items = page.get("items", [])
    check("返回 3 条", len(items) == 3, str(len(items)))
    check("total 与 health 一致", page.get("total") == ds.get("total"),
          f'{page.get("total")} vs {ds.get("total")}')
    if items:
        it = items[0]
        need = {"image_id", "file_name", "split", "width", "height"}
        check("字段齐全", need <= set(it), f"缺 {need - set(it)}")
        check("width/height 为正", it["width"] > 0 and it["height"] > 0,
              f'{it["width"]}x{it["height"]}')
        first_id = it["image_id"]

    # ids 批量模式
    st, b = get_json(f"/api/images?ids={first_id}")
    check("ids 模式只返回 1 条", len(b.get("items", [])) == 1, str(len(b.get("items", []))))
    check("ids 模式 id 对得上", b["items"][0]["image_id"] == first_id)

    # 分页
    st, p2 = get_json("/api/images?offset=1&limit=2")
    check("offset 生效", p2["items"][0]["image_id"] != first_id,
          p2["items"][0]["image_id"])

    # ---------------------------------------------------- file
    print("\n[4] GET /api/images/{id}/file")
    st, hdr, raw = http("GET", f"/api/images/{first_id}/file")
    check("200", st == 200, str(st))
    ctype = hdr.get("content-type", "")
    check("Content-Type 是图片", ctype.startswith("image/"), ctype)
    check("有 ETag", bool(hdr.get("etag")), hdr.get("etag", ""))
    check("有 Cache-Control", "max-age" in hdr.get("cache-control", ""),
          hdr.get("cache-control", "")[:40])
    etag = hdr.get("etag")

    st2, _, raw2 = http("GET", f"/api/images/{first_id}/file",
                        headers={"If-None-Match": etag})
    check("If-None-Match -> 304", st2 == 304, str(st2))

    st3, h3, raw3 = http("GET", f"/api/images/{first_id}/file",
                         headers={"Range": "bytes=0-99"})
    check("Range -> 206", st3 == 206, str(st3))
    check("Range 长度=100", len(raw3) == 100, str(len(raw3)))
    check("Content-Range 正确", h3.get("content-range", "").startswith("bytes 0-99/"),
          h3.get("content-range", ""))

    st4, _, _ = http("GET", "/api/images/__not_exist__/file")
    check("未知 id -> 404", st4 == 404, str(st4))

    # ---------------------------------------------------- predict
    print("\n[5] POST /api/predict")
    st, _, pr = http("POST", "/api/predict", {"image_id": first_id})
    check("200", st == 200, str(st))
    if st == 200:
        body = json.loads(pr.decode("utf-8"))
        check("有 instances 字段", "instances" in body)
        check("image_width/height 存在",
              body.get("image_width", 0) > 0 and body.get("image_height", 0) > 0,
              f'{body.get("image_width")}x{body.get("image_height")}')
        lat = body.get("latency_ms", 1e9)
        check("latency <= 300ms", lat <= 300, f"{lat} ms")
        check("model_version 存在", bool(body.get("model_version")))

        insts = body["instances"]
        print(f"        实例数 {len(insts)}，时延 {lat} ms")
        if insts:
            i0 = insts[0]
            need = {"class_id", "class_name", "score", "polygon", "bbox"}
            check("实例字段齐全", need <= set(i0), f"缺 {need - set(i0)}")
            poly = i0["polygon"]
            check("polygon 顶点 >= 3", len(poly) >= 3, str(len(poly)))
            check("polygon 归一化在 [0,1]",
                  all(0 <= p[0] <= 1 and 0 <= p[1] <= 1 for p in poly))
            check("polygon 顶点 <= 64", len(poly) <= 64, str(len(poly)))
            bb = i0["bbox"]
            check("bbox 4 个且有序",
                  len(bb) == 4 and bb[0] <= bb[2] and bb[1] <= bb[3], str(bb))
            check("score 在 (0,1]", 0 < i0["score"] <= 1, str(i0["score"]))

    # 缓存是否落盘（契约 §4.2）
    from pathlib import Path
    cache = Path(__file__).resolve().parents[2] / "out" / "pred_cache" / f"{first_id}.json"
    check("pred 缓存已落盘", cache.is_file(), str(cache.relative_to(cache.parents[2])))
    if cache.is_file():
        cd = json.loads(cache.read_text(encoding="utf-8"))
        check("缓存含 pred_ts", "pred_ts" in cd)
        check("缓存含 model_version", "model_version" in cd)

    # ---------------------------------------------------- 错误路径
    print("\n[6] 错误路径")
    st, _, _ = http("POST", "/api/predict", {"image_id": "__nope__"})
    check("未知 image_id -> 404", st == 404, str(st))
    st, _, _ = http("POST", "/api/predict", {})
    check("两个入参都缺 -> 422", st == 422, str(st))
    st, _, _ = http("POST", "/api/predict",
                  {"image_id": first_id, "image_base64": "AAAA"})
    check("两个入参都给 -> 422", st == 422, str(st))

    # ---------------------------------------------------- 提交标注
    print("\n[7] POST /api/annotations")
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[2]
    test_arm = "smoketest"
    test_arm_dir = root / "out" / "annotations" / test_arm
    if test_arm_dir.exists():
        for f in test_arm_dir.glob("*"):
            f.unlink()

    # 先对这张图跑一次 predict，确保 pred_cache 存在（否则测不到快照回填）
    http("POST", "/api/predict", {"image_id": first_id})

    poly = [[0.10, 0.10], [0.30, 0.10], [0.30, 0.40], [0.10, 0.40]]
    st, _, raw = http(
        "POST", f"/api/annotations?arm={test_arm}",
        {
            "image_id": first_id,
            "label_state": "labeled",
            "instances": [
                {"class_id": 0, "polygon": poly},
                {"class_id": 3, "polygon": [[0.5, 0.5], [0.6, 0.5], [0.6, 0.6]]},
            ],
            "elapsed_ms": 12345,
            "operator": "smoke",
            "session_id": "smoke-session",
        },
    )
    check("200", st == 200, str(st))
    if st == 200:
        r = json.loads(raw.decode("utf-8"))
        check("ok=true", r.get("ok") is True)
        check("n_instances=2", r.get("n_instances") == 2, str(r.get("n_instances")))
        check("arm 回显正确", r.get("arm") == test_arm, r.get("arm"))
        check("pred_cached=true（快照从缓存回填）", r.get("pred_cached") is True)
        check("label_state 回显", r.get("label_state") == "labeled")

        txt = root / r["label_path"]
        meta = txt.with_suffix(".json")
        check("标签 txt 已写", txt.is_file(), r["label_path"])
        check("元数据 json 已写", meta.is_file(), meta.name)

        if txt.is_file():
            body_lines = [ln for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
            check("txt 有 2 行（每实例一行）", len(body_lines) == 2, str(len(body_lines)))
            parts = body_lines[0].split()
            check("txt 每行 1+2n 个字段（class + 顶点）",
                  len(parts) == 1 + 2 * len(poly), str(len(parts)))
            check("txt 首列是 class_id", parts[0] == "0", parts[0])
            vals = [float(v) for v in parts[1:]]
            check("txt 坐标为归一化 [0,1]", all(0 <= v <= 1 for v in vals))

        if meta.is_file():
            m = json.loads(meta.read_text(encoding="utf-8"))
            check("meta.label_state 正确", m.get("label_state") == "labeled")
            check("meta 含 pred_snapshot", isinstance(m.get("pred_snapshot"), list))
            check("meta.pred_snapshot 非空（缓存命中）",
                  len(m.get("pred_snapshot") or []) > 0, str(len(m.get("pred_snapshot") or [])))
            check("meta 含 elapsed_ms / operator",
                  m.get("elapsed_ms") == 12345 and m.get("operator") == "smoke")

    # no_target：应落空 txt
    st, _, raw = http(
        "POST", f"/api/annotations?arm={test_arm}",
        {"image_id": first_id, "label_state": "no_target", "instances": []},
    )
    check("no_target 提交 200", st == 200, str(st))
    if st == 200:
        r = json.loads(raw.decode("utf-8"))
        t = root / r["label_path"]
        check("no_target 落空 txt", t.is_file() and t.stat().st_size == 0,
              f"size={t.stat().st_size if t.exists() else 'missing'}")

    # 校验：no_target 带实例 / 坐标越界 / class_id 越界 / 顶点太少
    st, _, _ = http("POST", f"/api/annotations?arm={test_arm}",
                    {"image_id": first_id, "label_state": "no_target",
                     "instances": [{"class_id": 0, "polygon": poly}]})
    check("no_target 带实例 -> 422", st == 422, str(st))
    st, _, _ = http("POST", f"/api/annotations?arm={test_arm}",
                    {"image_id": first_id, "label_state": "labeled",
                     "instances": [{"class_id": 0,
                                    "polygon": [[1.5, 0.1], [0.3, 0.1], [0.3, 0.4]]}]})
    check("坐标越界 -> 422", st == 422, str(st))
    st, _, _ = http("POST", f"/api/annotations?arm={test_arm}",
                    {"image_id": first_id, "label_state": "labeled",
                     "instances": [{"class_id": 99, "polygon": poly}]})
    check("class_id 越界 -> 422", st == 422, str(st))
    st, _, _ = http("POST", f"/api/annotations?arm={test_arm}",
                    {"image_id": first_id, "label_state": "labeled",
                     "instances": [{"class_id": 0, "polygon": [[0, 0], [1, 1]]}]})
    check("顶点少于 3 -> 422", st == 422, str(st))

    # ---------------------------------------------------- 操作日志
    print("\n[8] POST /api/logs/ops")
    ev = [
        {"ts": "2026-11-02 18:00:01", "image": "smoke_img", "op": "改类别",
         "instance": 0, "detail": "Apple -> Tomato", "n_ops_on_image": 1, "api_ms": "20.5"},
        {"ts": "2026-11-02 18:00:05", "image": "smoke_img", "op": "删实例",
         "instance": 1, "detail": "Tomato", "n_ops_on_image": 2, "api_ms": "20.5"},
        {"ts": "2026-11-02 18:00:09", "image": "smoke_img", "op": "整张确认",
         "instance": -1, "detail": "2 个实例", "n_ops_on_image": 3, "api_ms": ""},
    ]
    st, _, raw = http("POST", "/api/logs/ops", {"session_id": "smoke-session", "events": ev})
    check("200", st == 200, str(st))
    if st == 200:
        r = json.loads(raw.decode("utf-8"))
        check("appended=3", r.get("appended") == 3, str(r.get("appended")))
        check("返回 path 以 ops.csv 结尾", str(r.get("path", "")).endswith("ops.csv"),
              str(r.get("path")))
        csv_path = root / r["path"]
        check("ops.csv 已生成", csv_path.is_file())
        if csv_path.is_file():
            raw_bytes = csv_path.read_bytes()
            check("CSV 带 UTF-8 BOM（m5 用 utf-8-sig 读）", raw_bytes[:3] == b"\xef\xbb\xbf",
                  raw_bytes[:3].hex())
            text = raw_bytes.decode("utf-8-sig")
            rows = [ln for ln in text.splitlines() if ln.strip()]
            check("表头逐字对齐 m4_app.py:131",
                  rows[0] == "ts,image,op,instance,detail,n_ops_on_image,api_ms", rows[0])
            check("末尾三行是本次上报", len(rows) >= 4, f"共 {len(rows)} 行")
            check("中文 detail 未乱码", "Apple -> Tomato" in text)

    st, _, _ = http("POST", "/api/logs/ops", {
        "session_id": "s", "events": [{"ts": "t", "image": "i", "op": "瞎写一个", "instance": -1}],
    })
    check("词表外的 op -> 422", st == 422, str(st))
    st, _, _ = http("POST", "/api/logs/ops", {"session_id": "s", "events": []})
    check("空事件列表 -> 422", st == 422, str(st))

    # ---------------------------------------------------- 导出（P3 占位）
    print("\n[9] GET /api/logs/export")
    st, _, _ = http("GET", "/api/logs/export")
    check("返回 501（不是 404）", st == 501, str(st))
    st, _, _ = http("GET", "/api/logs/export?format=json")
    check("format=json -> 400", st == 400, str(st))

    # ---------------------------------------------------- 清理测试产物
    if test_arm_dir.exists():
        for f in test_arm_dir.glob("*"):
            f.unlink()
        test_arm_dir.rmdir()

    # ---------------------------------------------------- 汇总
    print("\n" + "=" * 64)
    total = passed + failed
    print(f"结果 {passed}/{total} 通过" + (f"，{failed} 失败" if failed else "，全部通过"))
    print("=" * 64)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
