"""交付前自测：重分片逻辑 + 换卡 + 卸载/挂回 + 失败保留旧发布。

运行: python3 tests/selftest.py
退出码 0 = 全部通过。
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import reshard  # noqa: E402
from reshard import (ARRAY_KEYS, MissingShardError, OverCapacityError,  # noqa: E402
                     SimulatedWriteError, StepMismatchError, split_sizes)
from sample_data import ensure_samples  # noqa: E402
from storage import ReshardService  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, fn) -> None:
    try:
        fn()
        results.append((PASS, name, ""))
        print(f"[PASS] {name}")
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc(limit=3)
        results.append((FAIL, name, tb))
        print(f"[FAIL] {name}\n{tb}")


def make_service(tmp: str) -> ReshardService:
    svc = ReshardService(Path(tmp))
    ensure_samples(svc.old)
    return svc


def load_source_arrays(data_dir: Path, ckpt_id: str):
    """直接从旧卷目录读源数组，用于和发布结果比对。"""
    ckpt = data_dir / "old_volume" / "checkpoints" / ckpt_id
    per_rank = {k: [] for k in ARRAY_KEYS}
    r = 0
    while (ckpt / f"rank_{r}.npz").is_file():
        with np.load(ckpt / f"rank_{r}.npz") as z:
            for k in ARRAY_KEYS:
                per_rank[k].append(z[k])
        r += 1
    return {k: np.concatenate(v) for k, v in per_rank.items()}


def published_arrays(svc: ReshardService):
    pub = svc.new.published
    out = {k: [] for k in ARRAY_KEYS}
    steps = []
    for r in range(pub["ranks"]):
        with np.load(svc.new.published_root / pub["id"] / f"rank_{r}.npz") as z:
            for k in ARRAY_KEYS:
                out[k].append(z[k])
            steps.append(int(z["step"]))
    return {k: np.concatenate(v) for k, v in out.items()}, steps


# ---------- 1. 纯逻辑：等分规则 ----------
def t_split_sizes():
    assert split_sizes(30, 5) == [6, 6, 6, 6, 6]
    assert split_sizes(10, 3) == [4, 3, 3], "余数应给低 rank"
    assert split_sizes(31, 4) == [8, 8, 8, 7]
    assert split_sizes(2, 5) == [1, 1, 0, 0, 0], "rank 多于元素时低 rank 先分"
    assert split_sizes(7, 1) == [7]
    assert sum(split_sizes(100, 9)) == 100


# ---------- 2. 换卡 3 -> 5：拼接一致 + 每片带训练步 ----------
def t_reshard_3_to_5():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        op = svc.run("ckpt-demo-3rank", 5, 1 << 20, True, False)
        assert op["status"] == "published", op
        assert op["shard_sizes"] == [6, 6, 6, 6, 6]
        src = load_source_arrays(Path(tmp), "ckpt-demo-3rank")
        got, steps = published_arrays(svc)
        for k in ARRAY_KEYS:
            assert np.array_equal(got[k], src[k]), f"{k} 拼接后与源不一致"
        assert steps == [42000] * 5, "每片都应带训练步"
        assert svc.new.published["step"] == 42000


# ---------- 3. 换卡 3 -> 2：余数给低 rank ----------
def t_reshard_3_to_2():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        op = svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        assert op["status"] == "published", op
        assert op["shard_sizes"] == [15, 15]
        src = load_source_arrays(Path(tmp), "ckpt-demo-3rank")
        got, _ = published_arrays(svc)
        for k in ARRAY_KEYS:
            assert np.array_equal(got[k], src[k])


# ---------- 4. 奇数总量余数分布 ----------
def t_remainder_to_low_ranks():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        op = svc.run("ckpt-demo-3rank", 7, 1 << 20, False, False)  # 30 = 7*4+2
        assert op["status"] == "published", op
        assert op["shard_sizes"] == [5, 5, 4, 4, 4, 4, 4], op["shard_sizes"]


# ---------- 5. 缺片：失败并保留上次发布 ----------
def t_missing_shard_keeps_published():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        ok = svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        assert ok["status"] == "published"
        before = svc.new.published["id"]
        bad = svc.run("ckpt-missing-rank", 4, 1 << 20, False, False)
        assert bad["status"] == "failed"
        assert "缺片" in bad["reason"], bad["reason"]
        assert svc.new.published["id"] == before, "失败后应保留上次发布"
        assert svc.new.staging_bytes() == 0, "失败后暂存应清理"


# ---------- 6. 训练步不一致：失败并保留上次发布 ----------
def t_step_mismatch_keeps_published():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        before = svc.new.published["id"]
        bad = svc.run("ckpt-step-mismatch", 4, 1 << 20, False, False)
        assert bad["status"] == "failed"
        assert "训练步不一致" in bad["reason"], bad["reason"]
        assert svc.new.published["id"] == before


# ---------- 7. 容量超限（上一套 + 本次暂存都计入） ----------
def t_over_capacity_keeps_published():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        before = svc.new.published["id"]
        prev_bytes = svc.new.published_bytes()
        # 容量只比上一套大一点，必然装不下上一套 + 本次暂存
        bad = svc.run("ckpt-demo-3rank", 4, prev_bytes + 10, False, False)
        assert bad["status"] == "failed"
        assert "容量超限" in bad["reason"], bad["reason"]
        assert svc.new.published["id"] == before
        assert svc.new.staging_bytes() == 0


# ---------- 8. 模拟写失败：保留上次发布，暂存清理 ----------
def t_simulated_write_failure():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        before = svc.new.published["id"]
        bad = svc.run("ckpt-demo-3rank", 6, 1 << 20, False, True)
        assert bad["status"] == "failed"
        assert "模拟写失败" in bad["reason"], bad["reason"]
        assert svc.new.published["id"] == before
        assert svc.new.staging_bytes() == 0


# ---------- 9. 峰值 = 上一套 + 本次暂存 ----------
def t_peak_accounting():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        op1 = svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        assert op1["op_peak"] == 0 + op1["staged_bytes"]
        peak1 = svc.new.peak_bytes
        op2 = svc.run("ckpt-demo-3rank", 5, 1 << 20, False, False)
        assert op2["op_peak"] == op2["prev_published_bytes"] + op2["staged_bytes"]
        assert op2["prev_published_bytes"] > 0, "第二套发布时上一套应计入"
        assert svc.new.peak_bytes == max(peak1, op2["op_peak"])


# ---------- 10. 卸载 -> 依赖旧卷的结果不可读；挂回同一卷恢复；挂不同卷不恢复 ----------
def t_unmount_dependency_cycle():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        op = svc.run("ckpt-demo-3rank", 4, 1 << 20, True, False)  # 依赖旧卷
        assert op["status"] == "published"
        assert svc.published_readable() is True
        svc.old.unmount()
        assert svc.published_readable() is False, "卸载后应不可读"
        svc.old.mount("oldvol-different-000")
        assert svc.published_readable() is False, "挂不同卷不应恢复"
        svc.old.mount()  # 挂回同一卷
        assert svc.published_readable() is True, "挂回同一卷应恢复"


# ---------- 11. 不依赖旧卷的结果：卸载后仍可读 ----------
def t_independent_result_survives_unmount():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        svc.run("ckpt-demo-3rank", 4, 1 << 20, False, False)  # 不依赖旧卷
        svc.old.unmount()
        assert svc.published_readable() is True


# ---------- 12. 卸载时源不可读，不能重分片 ----------
def t_unmounted_source_unreadable():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        svc.old.unmount()
        bad = svc.run("ckpt-demo-3rank", 4, 1 << 20, False, False)
        assert bad["status"] == "failed"
        assert "源检查点不可读" in bad["reason"], bad["reason"]
        assert svc.new.published is None


# ---------- 13. HTTP 冒烟：页面 / 重分片 / 下载 / 卸载后 409 / 挂回恢复 ----------
def t_http_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DATA_DIR"] = tmp
        # 延迟导入，让 server 用临时 DATA_DIR
        for mod in ("server",):
            sys.modules.pop(mod, None)
        import server  # noqa: WPS433

        c = server.app.test_client()
        assert c.get("/").status_code == 200

        r = c.post("/reshard", data={
            "ckpt_id": "ckpt-demo-3rank", "n_new": "5",
            "capacity": str(1 << 20), "can_mount_old": "on",
        }, follow_redirects=False)
        assert r.status_code == 302
        pub = server.service.new.published
        assert pub and pub["ranks"] == 5

        # 下载 rank_3 并核对内容 = 全局下标 [18, 24)
        resp = c.get("/download/3")
        assert resp.status_code == 200
        data = np.load(io.BytesIO(resp.data))
        src = load_source_arrays(Path(tmp), "ckpt-demo-3rank")
        for k in ARRAY_KEYS:
            assert np.array_equal(data[k], src[k][18:24]), f"下载分片 {k} 内容错误"
        assert int(data["step"]) == 42000

        # 卸载 -> 依赖旧卷，下载应 409
        c.post("/mount", data={"action": "unmount"})
        assert c.get("/download/3").status_code == 409
        page = c.get("/").data.decode()
        assert "不可读" in page

        # 挂回同一卷 -> 恢复
        c.post("/mount", data={"action": "mount_same"})
        assert c.get("/download/3").status_code == 200


# ---------- 崩溃恢复辅助 ----------
class ProcessDied(BaseException):
    """模拟进程在发布中途退出（不被 run 的 except Exception 捕获）。"""


def publish_old_set(tmp: str) -> tuple:
    """先发布一套 2 rank 旧发布，返回 (service, 旧发布 id, 旧套目录)。"""
    svc = make_service(tmp)
    op = svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
    assert op["status"] == "published", op
    old_id = svc.new.published["id"]
    old_dir = svc.new.published_root / old_id
    assert old_dir.is_dir()
    return svc, old_id, old_dir


def restart_service(tmp: str):
    """模拟进程重启：重新构造服务，构造即触发 recover()。"""
    svc2 = ReshardService(Path(tmp))
    ensure_samples(svc2.old)
    return svc2


def assert_page_and_downloads_match(svc: ReshardService) -> None:
    """页面发布编号、发布记录与每个 rank 的下载文件必须指向同一套且文件齐全。"""
    pub = svc.new.published
    assert pub is not None, "恢复后必须存在一套发布"
    set_dir = svc.new.published_root / pub["id"]
    assert set_dir.is_dir(), f"发布记录指向的目录 {set_dir} 不存在"
    for r in range(int(pub["ranks"])):
        f = set_dir / f"rank_{r}.npz"
        assert f.is_file(), f"{pub['id']} 的 rank_{r}.npz 缺失，下载会 404"
    # published 下只允许存在这一套，杜绝页面与下载指向不同套
    siblings = [p for p in svc.new.published_root.iterdir() if p.is_dir()]
    assert siblings == [set_dir], siblings
    assert svc.new.staging_bytes() == 0, "恢复后暂存区应清空"


def fresh_client(tmp: str):
    os.environ["DATA_DIR"] = tmp
    sys.modules.pop("server", None)
    import server  # noqa: WPS433
    return server, server.app.test_client()


def stage_verified_set(svc: ReshardService, n_new: int):
    """跑到“全套回读校验通过、待发布标记已写”为止，不执行发布移动。

    返回 (暂存目录, 发布 id)，用于手工布置替换中途的崩溃现场。
    """
    import storage as storage_mod
    with svc.lock:
        svc.new.reclaim_leftovers()
        stg = svc.new.new_staging_dir()
        src = reshard.load_source_checkpoint(
            svc.old.checkpoints_dir() / "ckpt-demo-3rank")
        shards = {k: reshard.reshard_arrays(src.arrays[k], n_new)
                  for k in reshard.ARRAY_KEYS}
        sizes = reshard.split_sizes(src.total, n_new)
        files = []
        for r in range(n_new):
            p = stg / f"rank_{r}.npz"
            reshard.write_shard(p, {k: shards[k][r] for k in reshard.ARRAY_KEYS},
                                src.step)
            files.append(p)
        reshard.verify_shards(files, sizes, src.step, src)
        import uuid
        from datetime import datetime
        pub_id = f"pub-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        record = {
            "id": pub_id, "source_ckpt": src.ckpt_id, "step": src.step,
            "ranks": n_new, "total_elems": src.total, "dtype": src.dtype,
            "shard_sizes": sizes, "bytes": sum(f.stat().st_size for f in files),
            "requires_old_volume": False, "old_volume_id": None,
            "created_at": "test",
        }
        storage_mod._atomic_write_json(stg / storage_mod.MANIFEST_NAME, record)
        storage_mod._atomic_write_json(svc.new.pending_file, {
            "id": pub_id, "staging": stg.name, "created_at": "test",
        })
        return stg, pub_id


def make_mid_swap(svc: ReshardService, n_new: int):
    """布置替换中途现场：旧套在备份目录、新套在暂存、pending 存在、记录仍旧。"""
    old_id = svc.new.published["id"]
    stg, new_id = stage_verified_set(svc, n_new)
    backup = svc.new._new_backup_dir()
    # 与 publish() 的真实动作一致：整个 published/ 重命名为 .backup-xxx，
    # 旧套因此位于 .backup-xxx/<旧id>/（单层嵌套）
    shutil.move(str(svc.new.published_root), str(backup))
    svc.new.published_root.mkdir(parents=True)
    assert svc.new.pending_file.is_file()
    assert svc.new.published["id"] == old_id  # state 里的记录还没更新
    return stg, new_id, old_id, backup


# ---------- 14. 回读校验前中断（暂存写一半）：重启恢复上次发布 ----------
def t_crash_before_verify_restores_old():
    with tempfile.TemporaryDirectory() as tmp:
        svc, old_id, old_dir = publish_old_set(tmp)

        # 模拟在写入第 2 个分片时进程退出：暂存只留下不完整的一套，无待发布标记
        with svc.lock:
            svc.new.reclaim_leftovers()
            stg = svc.new.new_staging_dir()
            src = reshard.load_source_checkpoint(
                svc.old.checkpoints_dir() / "ckpt-demo-3rank")
            shards = {k: reshard.reshard_arrays(src.arrays[k], 5)
                      for k in reshard.ARRAY_KEYS}
            reshard.write_shard(
                stg / "rank_0.npz",
                {k: shards[k][0] for k in reshard.ARRAY_KEYS}, src.step)
            assert not svc.new.pending_file.is_file()

        svc2 = restart_service(tmp)
        assert svc2.new.published is not None
        assert svc2.new.published["id"] == old_id, "未通过全套校验必须恢复上次发布"
        assert old_dir.is_dir()
        assert_page_and_downloads_match(svc2)
        assert not any(svc2.new.dir.glob(".backup-*")), "备份遗留应被清理"
        assert not any(svc2.new.staging_root.iterdir())


# ---------- 15. 分片已替换、发布记录还没保存（移动中途）：重启不再 404 ----------
def t_crash_mid_swap_after_verified_promotes_new():
    with tempfile.TemporaryDirectory() as tmp:
        svc, old_id, _ = publish_old_set(tmp)
        stg, new_id, _, _ = make_mid_swap(svc, 5)
        assert stg.is_dir()
        # 旧套此刻位于备份目录内
        assert any(svc.new.dir.glob(".backup-*"))

        svc2 = restart_service(tmp)
        assert svc2.new.published is not None, "必须恢复出一套完整套"
        assert svc2.new.published["id"] == new_id, "校验已通过的新套应被发布"
        assert (svc2.new.published_root / new_id).is_dir()
        assert not (svc2.new.published_root / old_id).exists()
        assert_page_and_downloads_match(svc2)
        assert not any(svc2.new.dir.glob(".backup-*"))


# ---------- 16. 发布成功提示已可给出（state 已保存）后退出：保留新套 ----------
def t_crash_after_state_saved_keeps_new():
    with tempfile.TemporaryDirectory() as tmp:
        svc, old_id, _ = publish_old_set(tmp)
        op = svc.run("ckpt-demo-3rank", 4, 1 << 20, False, False)
        assert op["status"] == "published"
        new_id = svc.new.published["id"]
        assert new_id != old_id
        # 此时 pending 已清除、备份已删；再“重启”一次必须原样保留新套
        svc2 = restart_service(tmp)
        assert svc2.new.published["id"] == new_id
        assert_page_and_downloads_match(svc2)


# ---------- 17. 校验通过但新套意外损坏、旧备份完整：回退任一完整套（旧套） ----------
def t_crash_pending_new_broken_falls_back_to_old():
    with tempfile.TemporaryDirectory() as tmp:
        svc, old_id, _ = publish_old_set(tmp)
        stg, new_id, _, _ = make_mid_swap(svc, 5)
        # 把新套先落入 published，再破坏其中一个分片；旧套完整备份保留
        target = svc.new.published_root / new_id
        shutil.move(str(stg), str(target))
        (target / "rank_4.npz").unlink()
        assert svc.new.pending_file.is_file()

        svc2 = restart_service(tmp)
        assert svc2.new.published is not None, "两套都在时必须恢复出完整套"
        assert svc2.new.published["id"] == old_id, "新套不完整应回退完整旧套"
        assert (svc2.new.published_root / old_id).is_dir()
        assert_page_and_downloads_match(svc2)


# ---------- 18. 遗留文件存在时页面可打开、各 rank 可下载当前套 ----------
def t_leftovers_page_open_and_downloads_work():
    with tempfile.TemporaryDirectory() as tmp:
        svc, old_id, _ = publish_old_set(tmp)
        # 手工撒入各种遗留：半截暂存、备份目录、临时文件
        junk_stg = svc.new.staging_root / "stg-junk"
        junk_stg.mkdir()
        (junk_stg / "rank_0.npz").write_bytes(b"junk")
        backup = svc.new.dir / ".backup-deadbeef"
        backup.mkdir()
        (svc.new.dir / "state.json.tmp").write_text("{broken", encoding="utf-8")

        server_mod, client = fresh_client(tmp)  # 构造过程即完成恢复
        page = client.get("/")
        assert page.status_code == 200
        pub = server_mod.service.new.published
        assert pub["id"] == old_id
        for r in range(int(pub["ranks"])):
            resp = client.get(f"/download/{r}")
            assert resp.status_code == 200, f"rank {r} 下载失败: {resp.status_code}"
            assert resp.headers["Content-Disposition"].startswith(
                "attachment;"), "下载应带附件文件名"
            assert old_id in resp.headers["Content-Disposition"], "下载须指向当前套"
        # 越界 rank 仍然 404
        assert client.get("/download/99").status_code == 404


# ---------- 19. 下次重分片先释放遗留占用，再按原规则判断容量 ----------
def t_next_reshard_reclaims_leftover_capacity():
    with tempfile.TemporaryDirectory() as tmp:
        svc, old_id, _ = publish_old_set(tmp)
        # 制造一个占空间的遗留暂存
        junk = svc.new.staging_root / "stg-junk"
        junk.mkdir()
        for i in range(3):
            (junk / f"rank_{i}.npz").write_bytes(b"x" * 4096)
        junk_bytes = svc.new.staging_bytes()
        assert junk_bytes > 0
        # 遗留若不先释放，当前占用已经超过容量，操作在任何写入前就该失败
        prev_pub = svc.new.published_bytes()
        cap = prev_pub + 4000  # 够装旧套 + 3-rank 新暂存（约 3360 B）
        assert svc.new.usage_bytes() > cap

        # 容量装得下“上一套 + 新暂存”，但装不下“上一套 + 新暂存 + 遗留”：
        # 先释放遗留后按原规则判断，应当发布成功
        op = svc.run("ckpt-demo-3rank", 3, cap, False, False)
        assert op["status"] == "published", op
        assert op["prev_published_bytes"] + op["staged_bytes"] <= cap
        assert svc.new.staging_bytes() == 0
        assert svc.new.published["id"] != old_id
        assert not any(svc.new.dir.glob(".backup-*"))

        # 反过来：释放遗留后仍装不下新暂存时，照旧超额失败并保留当前发布
        cur_id = svc.new.published["id"]
        tight = svc.new.published_bytes() + 10
        bad = svc.run("ckpt-demo-3rank", 6, tight, False, False)
        assert bad["status"] == "failed"
        assert "容量超限" in bad["reason"]
        assert svc.new.published["id"] == cur_id
        assert svc.new.staging_bytes() == 0, "失败后暂存与遗留都应清空"


# ---------- 20. 恢复不读取 NPZ 正文、不复制分片（只做元数据判断与 rename/删除） ----------
def t_recovery_never_touches_npz_payloads():
    import storage as storage_mod

    with tempfile.TemporaryDirectory() as tmp:
        svc, old_id, _ = publish_old_set(tmp)
        make_mid_swap(svc, 5)  # 替换中途现场：新套在暂存、旧套在备份、pending 存在

        real_np_load = np.load
        blocked = {"load": 0, "direct_copy": 0}

        def blocking_load(*a, **kw):  # 恢复路径一旦读 NPZ 正文即失败
            blocked["load"] += 1
            raise AssertionError("恢复流程禁止读取 NPZ 正文")

        real_copy = shutil.copy
        real_copy2 = shutil.copy2
        real_copytree = shutil.copytree
        real_move = shutil.move

        def called_from_move():
            """仅放行 shutil.move 跨设备回退时的内部复制（语义仍是 rename）。"""
            frame = sys._getframe()
            while frame is not None:
                code = frame.f_code
                if code.co_name in ("move", "_move"):
                    return True
                frame = frame.f_back
            return False

        def blocking_copy(*a, **kw):
            if called_from_move():
                return real_copy(*a, **kw)
            blocked["direct_copy"] += 1
            raise AssertionError("恢复流程禁止直接复制分片")

        def blocking_copy2(*a, **kw):
            if called_from_move():
                return real_copy2(*a, **kw)
            blocked["direct_copy"] += 1
            raise AssertionError("恢复流程禁止直接复制分片")

        def blocking_copytree(*a, **kw):
            if called_from_move():
                return real_copytree(*a, **kw)
            blocked["direct_copy"] += 1
            raise AssertionError("恢复流程禁止直接复制分片树")

        np.load = blocking_load
        shutil.copy = blocking_copy
        shutil.copy2 = blocking_copy2
        shutil.copytree = blocking_copytree
        try:
            # 与 ReshardService 一致：传入数据卷根目录，NewVolume 内部再拼 new_volume
            vol = storage_mod.NewVolume(Path(tmp))
        finally:
            np.load = real_np_load
            shutil.copy = real_copy
            shutil.copy2 = real_copy2
            shutil.copytree = real_copytree
        assert blocked["load"] == 0 and blocked["direct_copy"] == 0, blocked
        assert vol.published is not None
        # 恢复后真正下载仍可正常读出正文
        with real_np_load(vol.published_root / vol.published["id"] / "rank_0.npz") as z:
            assert "param" in z.files and int(z["step"]) == 42000
        # 生产代码只允许 rename 级移动（shutil.move），不允许直接复制 API
        source = Path(storage_mod.__file__).read_text(encoding="utf-8")
        assert "shutil.copytree" not in source
        assert "shutil.copy(" not in source and "shutil.copy2(" not in source
        assert real_move is shutil.move, "恢复不得改写 move 语义"


# ---------- 21. 连续多次重启：恢复幂等，发布套保持稳定 ----------
def t_recovery_is_idempotent_across_restarts():
    with tempfile.TemporaryDirectory() as tmp:
        svc, old_id, _ = publish_old_set(tmp)
        make_mid_swap(svc, 5)  # 替换中途现场
        first = restart_service(tmp)
        new_id = first.new.published["id"]
        assert new_id != old_id
        assert_page_and_downloads_match(first)
        # 再连续重启两次，结果必须完全一致，且不产生新遗留
        for _ in range(2):
            again = restart_service(tmp)
            assert again.new.published["id"] == new_id
            assert_page_and_downloads_match(again)
            assert not list(again.new.dir.glob(".backup-*"))
            assert not again.new.pending_file.is_file()


# ---------- 22. 崩溃恢复后走 HTTP：页面编号与每个 rank 下载指向同一套 ----------
def t_http_after_crash_recovery_consistent():
    with tempfile.TemporaryDirectory() as tmp:
        svc, old_id, _ = publish_old_set(tmp)
        make_mid_swap(svc, 5)  # 重启前：分片替换到一半、记录还是旧套
        server_mod, client = fresh_client(tmp)  # 构造服务即完成恢复
        page = client.get("/")
        assert page.status_code == 200
        pub = server_mod.service.new.published
        new_id = pub["id"]
        assert new_id != old_id
        html = page.data.decode()
        assert new_id in html, "页面发布编号必须指向恢复出的新套"
        assert old_id not in html, "页面不应再出现旧套编号"
        # 每个 rank 都能下载，且下载文件名属于新套
        for r in range(int(pub["ranks"])):
            resp = client.get(f"/download/{r}")
            assert resp.status_code == 200
            assert new_id in resp.headers["Content-Disposition"]
        # 下载内容回读与源全局拼接一致
        src = load_source_arrays(Path(tmp), "ckpt-demo-3rank")
        resp0 = client.get("/download/0")
        d0 = np.load(io.BytesIO(resp0.data))
        assert np.array_equal(d0["param"], src["param"][0:6])
        assert int(d0["step"]) == 42000
        # 旧卷挂载规则不变：卸载依赖旧卷的发布仍返回 409
        server_mod.service.new._state["published"]["requires_old_volume"] = True
        server_mod.service.new._state["published"]["old_volume_id"] = \
            server_mod.service.old.stored_id()
        server_mod.service.new.save()
        client.post("/mount", data={"action": "unmount"})
        assert client.get("/download/0").status_code == 409
        client.post("/mount", data={"action": "mount_same"})
        assert client.get("/download/0").status_code == 200


def main() -> int:
    tests = [
        ("等分规则（余数给低 rank）", t_split_sizes),
        ("换卡 3->5：拼接一致、每片带训练步", t_reshard_3_to_5),
        ("换卡 3->2：连续等分", t_reshard_3_to_2),
        ("余数分布 30->7 片", t_remainder_to_low_ranks),
        ("缺片：失败并保留上次发布", t_missing_shard_keeps_published),
        ("训练步不一致：失败并保留上次发布", t_step_mismatch_keeps_published),
        ("容量超限：失败并保留上次发布", t_over_capacity_keeps_published),
        ("模拟写失败：保留上次发布、清理暂存", t_simulated_write_failure),
        ("峰值 = 上一套 + 本次暂存", t_peak_accounting),
        ("卸载/挂不同卷/挂回同一卷的可读性", t_unmount_dependency_cycle),
        ("不依赖旧卷的结果卸载后仍可读", t_independent_result_survives_unmount),
        ("卸载时源不可读，重分片失败", t_unmounted_source_unreadable),
        ("HTTP 冒烟：页面/发布/下载/409/恢复", t_http_smoke),
        ("校验前中断：重启恢复上次发布、无 404", t_crash_before_verify_restores_old),
        ("替换中途崩溃：记录未保存也能恢复新套", t_crash_mid_swap_after_verified_promotes_new),
        ("已提示成功后退出：重启保留新套", t_crash_after_state_saved_keeps_new),
        ("校验通过但新套损坏：回退完整旧套", t_crash_pending_new_broken_falls_back_to_old),
        ("遗留文件存在时页面可开、下载当前套", t_leftovers_page_open_and_downloads_work),
        ("下次重分片先释放遗留再判容量", t_next_reshard_reclaims_leftover_capacity),
        ("恢复不读 NPZ 正文、不复制分片", t_recovery_never_touches_npz_payloads),
        ("连续多次重启恢复幂等", t_recovery_is_idempotent_across_restarts),
        ("崩溃恢复后 HTTP 页面与下载指向同一套", t_http_after_crash_recovery_consistent),
    ]
    for name, fn in tests:
        check(name, fn)
    failed = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
