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

    # ---------------------------------------------------- 汇总
    print("\n" + "=" * 64)
    total = passed + failed
    print(f"结果 {passed}/{total} 通过" + (f"，{failed} 失败" if failed else "，全部通过"))
    print("=" * 64)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
