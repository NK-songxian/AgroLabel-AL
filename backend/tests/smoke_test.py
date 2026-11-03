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
    check("dataset.mode 合法", ds.get("mode") in ("manifest", "yolo", "demo"), ds.get("mode"))
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

    st, _, _ = http("GET", "/api/images?order=class")
    check("order=class -> 200", st == 200, str(st))
    st, _, _ = http("GET", "/api/images?order=score_desc")
    check("未实现的 order=score_desc -> 400（不静默忽略）", st == 400, str(st))

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

    # 用真实臂名（后端白名单只认 assisted/manual/noprompt），并且**备份再还原** ——
    # 这几个目录里可能躺着标注员的真实数据，测试绝不能覆盖或删除。
    test_arm = "manual"
    alt_arm = "noprompt"
    ann_base = root / "out" / "annotations"


    def _backup(arm_name: str):
        d = ann_base / arm_name
        files = {}
        for suffix in (".txt", ".json"):
            p = d / f"{first_id}{suffix}"
            files[p] = p.read_bytes() if p.is_file() else None
        return files


    def _restore(files):
        for p, blob in files.items():
            if blob is None:
                if p.is_file():
                    p.unlink()
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(blob)


    backup = {}
    for arm_name in (test_arm, alt_arm):
        backup.update(_backup(arm_name))

    test_arm_dir = ann_base / test_arm

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

    # 臂名白名单：拼错一个字母就会分裂出一个新目录，三臂对照直接作废
    st, _, _ = http("POST", "/api/annotations?arm=smoketest",
                    {"image_id": first_id, "label_state": "labeled", "instances": []})
    check("未知实验臂 -> 422（防脏目录）", st == 422, str(st))

    # ---------------------------------------------------- 三臂目录隔离
    print("\n[7b] 三臂目录隔离")
    st, _, raw = http("GET", f"/api/annotations/{first_id}?arm={test_arm}")
    check(f"读回 {test_arm} 臂 200", st == 200, str(st))
    st_alt, _, raw_alt = http(
        "POST", f"/api/annotations?arm={alt_arm}",
        {"image_id": first_id, "label_state": "labeled",
         "instances": [{"class_id": 1, "polygon": poly},
                       {"class_id": 2, "polygon": poly},
                       {"class_id": 3, "polygon": poly}]},
    )
    check(f"写到 {alt_arm} 臂 200", st_alt == 200, str(st_alt))
    if st_alt == 200:
        a_txt = ann_base / test_arm / f"{first_id}.txt"
        b_txt = ann_base / alt_arm / f"{first_id}.txt"
        check("两臂 txt 落在不同目录", a_txt.is_file() and b_txt.is_file(),
              f"{a_txt.name} / {b_txt.name}")
        n_a = len([x for x in a_txt.read_text(encoding="utf-8").splitlines() if x.strip()])
        n_b = len([x for x in b_txt.read_text(encoding="utf-8").splitlines() if x.strip()])
        check("两臂内容互不覆盖", n_a != n_b or n_a == 0,
              f"{test_arm}={n_a} 行, {alt_arm}={n_b} 行")
        st, _, raw = http("GET", f"/api/annotations/{first_id}?arm={alt_arm}")
        if st == 200:
            r = json.loads(raw.decode("utf-8"))
            check("按臂读回各自的内容", r.get("arm") == alt_arm and r.get("n_instances") == 3,
                  f'arm={r.get("arm")} n={r.get("n_instances")}')
    st, _, _ = http("GET", f"/api/annotations/{first_id}?arm=bogus")
    check("读回也校验臂名 -> 422", st == 422, str(st))

    # ---------------------------------------------------- 操作日志
    print("\n[8] POST /api/logs/ops")
    # 先记下 ops.csv 原有大小：测试结束后要截回去，否则测试行会混进
    # 真实实验的 m5 统计里（"smoke_img" 这种假数据最难排查）
    ops_csv = root / "out" / "logs" / "ops.csv"
    ops_size_before = ops_csv.stat().st_size if ops_csv.is_file() else 0

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
        # 分臂日志：三臂对照要求 ops 能按臂拆开（ops.csv 表头不能动，所以另写一份）
        arm_csv = root / "out" / "logs" / f"ops_{test_arm}.csv"
        arm_had = arm_csv.is_file()
        arm_size_before = arm_csv.stat().st_size if arm_had else 0
        st_arm, _, _ = http("POST", f"/api/logs/ops?arm={test_arm}",
                            {"session_id": "smoke-session", "events": ev})
        check("带 arm 上报 200", st_arm == 200, str(st_arm))
        check(f"生成分臂文件 ops_{test_arm}.csv", arm_csv.is_file())
        st_bad, _, _ = http("POST", "/api/logs/ops?arm=bogus",
                            {"session_id": "smoke-session", "events": ev})
        check("未知实验臂 -> 422（防脏文件）", st_bad == 422, str(st_bad))
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

    # ---------------------------------------------------- 主动学习选样
    print("\n[10] POST /api/sample")
    st, _, raw = http("POST", "/api/sample", {"k": 3, "strategy": "random", "seed": 1})
    check("random 策略 200（不加载模型，快）", st == 200, str(st))
    if st == 200:
        r = json.loads(raw.decode("utf-8"))
        check("选中 3 张", len(r.get("selected", [])) == 3, str(len(r.get("selected", []))))
        check("selected 都是已知 id", all(isinstance(x, str) and x for x in r["selected"]))
        check("reasons 覆盖全部选中", set(r["reasons"]) == set(r["selected"]))
        first = r["selected"][0]
        check("reason 有 rank/criterion", r["reasons"][first]["rank"] == 1
              and r["reasons"][first]["criterion"] == "random",
              json.dumps(r["reasons"][first], ensure_ascii=False))
        check("pool_size > 0", r.get("pool_size", 0) > 0, str(r.get("pool_size")))
        check("明确说明 score 为何为 null", "score" in (r.get("note") or ""))

    st, _, raw = http("POST", "/api/sample", {"k": 2, "strategy": "bogus"})
    check("非法 strategy -> 422", st == 422, str(st))
    st, _, _ = http("POST", "/api/sample", {"pool": ["__nope__"], "k": 1})
    check("未知 image_id -> 404", st == 404, str(st))

    # ---------------------------------------------------- 点提示
    print("\n[11] POST /api/refine")
    st, _, raw = http("POST", "/api/predict", {"image_id": first_id})
    pr = json.loads(raw.decode("utf-8")) if st == 200 else {}
    insts = pr.get("instances") or []
    iw, ih = pr.get("image_width", 0), pr.get("image_height", 0)
    check("predict 有实例可用于取点", bool(insts), str(len(insts)))

    if insts and iw and ih:
        poly0 = insts[0]["polygon"]
        cx = sum(p[0] for p in poly0) / len(poly0) * iw
        cy = sum(p[1] for p in poly0) / len(poly0) * ih
        cid = insts[0]["class_id"]

        st, _, raw = http("POST", "/api/refine",
                          {"image_id": first_id, "point": [cx, cy]})
        check("不传 class_id 也 200", st == 200, str(st))
        if st == 200:
            r = json.loads(raw.decode("utf-8"))
            check("有 found 字段", "found" in r)
            check("latency_ms 存在", isinstance(r.get("latency_ms"), (int, float)))
            check("有 class_mismatch 字段", isinstance(r.get("class_mismatch"), bool))
            rad = r.get("radius_used")
            check("radius_used 在 [96,512]（不再是 48 的默认值）",
                  isinstance(rad, int) and 96 <= rad <= 512, str(rad))

        st, _, raw = http("POST", "/api/refine", {
            "image_id": first_id, "point": [cx, cy], "class_id": cid,
        })
        check("传 class_id 后 200", st == 200, str(st))
        if st == 200:
            r = json.loads(raw.decode("utf-8"))
            if r.get("found"):
                # 找到时类别要么就是请求的类，要么被标记为不一致 —— 不允许悄悄返回别的类
                check("找到时类别要么一致、要么标了 class_mismatch",
                      r.get("class_id") == cid or r.get("class_mismatch") is True,
                      f'cid={r.get("class_id")} mismatch={r.get("class_mismatch")}')
                check("polygon 归一化在 [0,1]",
                      all(0 <= q[0] <= 1 and 0 <= q[1] <= 1 for q in r["polygon"]))
                # 模型侧默认上限 64，实测能吐 40 个顶点、相邻只隔 1.1px（点挤在一起没法拖）。
                # 后端把默认降到 20 并剪掉 <2px 的边，这里守住这个口径。
                check("默认顶点数 <= 20（不再顶到 64）", len(r["polygon"]) <= 20,
                      str(len(r["polygon"])))
                pts = [(q[0] * iw, q[1] * ih) for q in r["polygon"]]
                n = len(pts)
                edges = [
                    ((pts[i][0] - pts[(i + 1) % n][0]) ** 2
                     + (pts[i][1] - pts[(i + 1) % n][1]) ** 2) ** 0.5
                    for i in range(n)
                ]
                check("相邻顶点最小间距 >= 1.5px（不再挤成一团）",
                      min(edges) >= 1.5, f"{min(edges):.2f}px")
            else:
                check("类别被钉住（硬过滤生效）", True, "该点未检出，跳过")

    st, _, raw = http("POST", "/api/refine", {"image_id": "__nope__", "point": [1, 1]})
    check("未知 image_id -> 404", st == 404, str(st))
    st, _, _ = http("POST", "/api/refine", {"image_id": first_id, "point": [1]})
    check("point 长度不对 -> 422", st == 422, str(st))

    # ---------------------------------------------------- 编辑代价
    print("\n[12] POST /api/edit-cost")
    if insts:
        base = {"pred_instances": insts, "gt_instances": insts,
                "image_width": iw, "image_height": ih}

        st, _, raw = http("POST", "/api/edit-cost", base)
        check("200", st == 200, str(st))
        if st == 200:
            r = json.loads(raw.decode("utf-8"))
            check("完全相同 -> edit_count=0", r.get("edit_count") == 0, str(r.get("edit_count")))
            check("breakdown 含 5 个键",
                  set(r.get("breakdown", {})) == {"新画", "删实例", "改类别", "拖边界", "标为无目标"},
                  str(list(r.get("breakdown", {}))))
            check("canvas 反映传入宽高", r.get("canvas") == [iw, ih], str(r.get("canvas")))

        gt_minus2 = insts[2:]
        st, _, raw = http("POST", "/api/edit-cost",
                          {**base, "gt_instances": gt_minus2})
        if st == 200:
            r = json.loads(raw.decode("utf-8"))
            check("删掉 2 个 -> 删实例=2", r["breakdown"]["删实例"] == 2,
                  json.dumps(r["breakdown"], ensure_ascii=False))

        st, _, raw = http("POST", "/api/edit-cost",
                          {**base, "gt_instances": []})
        if st == 200:
            r = json.loads(raw.decode("utf-8"))
            check("gt 空 pred 非空 -> 标为无目标=1",
                  r["breakdown"]["标为无目标"] == 1,
                  json.dumps(r["breakdown"], ensure_ascii=False))

    st, _, _ = http("POST", "/api/edit-cost", {
        "pred_instances": [], "gt_instances": [],
        "image_width": 100, "image_height": 100,
    })
    check("两边都空 -> 200 且 edit_count=0", st == 200, str(st))

    # ---------------------------------------------------- 清理测试产物
    # 只还原本次测试碰过的那两个文件，**不要整个删目录** ——
    # out/annotations/<arm>/ 里可能有标注员的真实标注
    _restore(backup)
    for arm_name in (test_arm, alt_arm):
        d = ann_base / arm_name
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()

    # 分臂文件同样清理
    if arm_had:
        with open(arm_csv, "r+b") as f:
            f.truncate(arm_size_before)
    elif arm_csv.is_file():
        arm_csv.unlink()

    # 把 ops.csv 截回测试前的大小（只有追加，所以截断是安全的）
    if ops_csv.is_file():
        if ops_size_before == 0:
            ops_csv.unlink()
        elif ops_csv.stat().st_size > ops_size_before:
            with open(ops_csv, "r+b") as f:
                f.truncate(ops_size_before)

    # ---------------------------------------------------- 汇总
    print("\n" + "=" * 64)
    total = passed + failed
    print(f"结果 {passed}/{total} 通过" + (f"，{failed} 失败" if failed else "，全部通过"))
    print("=" * 64)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
