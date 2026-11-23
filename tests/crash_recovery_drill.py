"""真实 SIGKILL 崩溃恢复演练（独立脚本，不计入 selftest）。

用两个子进程操作同一个数据卷：
  1. 进程 A：正常发布一套 2-rank；
  2. 进程 B：monkeypatch 让 publish 在“旧套已挪备份、新套移动前”被杀（SIGKILL）；
  3. 进程 C：重新构造服务（模拟重启），断言恢复出新套且页面/下载一致。
全程用独立目录与 -I 隔离运行。
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
PYLIBS = "/tmp/pylibs/lib/python3.11/site-packages"


def run_worker(data_dir: str, stage: str) -> subprocess.CompletedProcess:
    code = f"""
import os, sys, signal
sys.path.insert(0, {str(APP)!r})
sys.path.insert(0, {PYLIBS!r})
import storage, shutil
from sample_data import ensure_samples
from storage import ReshardService

svc = ReshardService({data_dir!r})
ensure_samples(svc.old)

if {stage!r} == "publish_old":
    op = svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
    assert op["status"] == "published", op
    print("OLD_ID=" + svc.new.published["id"])

elif {stage!r} == "crash_mid_swap":
    # 回读校验与 pending 落盘后，publish() 第一次 move（把 published 挪成备份）
    # 的瞬间 SIGKILL：分片已在暂存、旧套已挪走、发布记录尚未更新
    import shutil as _sh
    real_move = _sh.move
    def killing_move(src, dst, *a, **k):
        result = real_move(src, dst, *a, **k)  # 先完成 published -> backup 的移动
        if str(src).endswith("published") and str(dst).count(".backup-"):
            os.kill(os.getpid(), signal.SIGKILL)  # 旧套已挪走、新套未动时被杀
        return result
    _sh.move = killing_move
    try:
        svc.run("ckpt-demo-3rank", 5, 1 << 20, False, False)
    except BaseException as e:
        print("worker ended:", type(e).__name__)

elif {stage!r} == "crash_before_verify":
    # 在写入第 2 个分片时 SIGKILL：暂存不完整、没有 pending 标记。
    # storage 里是 from reshard import write_shard，必须 patch 已绑定的引用。
    real_write = storage.write_shard
    def killing_write(path, arrays, step):
        n = int(str(path).split("rank_")[1].split(".")[0])
        if n == 1:
            os.kill(os.getpid(), signal.SIGKILL)
        return real_write(path, arrays, step)
    storage.write_shard = killing_write
    try:
        svc.run("ckpt-demo-3rank", 5, 1 << 20, False, False)
    except BaseException as e:
        print("worker ended:", type(e).__name__)

elif {stage!r} == "restart":
    # 构造即恢复；校验页面记录与磁盘布局一致
    pub = svc.new.published
    assert pub is not None, "恢复后必须有发布"
    set_dir = svc.new.published_root / pub["id"]
    assert set_dir.is_dir()
    for r in range(int(pub["ranks"])):
        f = set_dir / ("rank_" + str(r) + ".npz")
        assert f.is_file(), "rank " + str(r) + " 缺失会 404"
    assert svc.new.staging_bytes() == 0
    assert not list(svc.new.dir.glob(".backup-*"))
    assert not svc.new.pending_file.is_file()
    print("RECOVERED_ID=" + pub["id"], "RANKS=" + str(pub["ranks"]))
"""
    env = dict(os.environ, PYTHONPATH=f"{PYLIBS}:{APP}")
    return subprocess.run([sys.executable, "-I", "-c", code],
                          capture_output=True, text=True, env=env)


def main() -> int:
    with tempfile.TemporaryDirectory() as data_dir:
        r1 = run_worker(data_dir, "publish_old")
        assert r1.returncode == 0, r1.stderr
        old_id = [l.split("=", 1)[1] for l in r1.stdout.splitlines()
                  if l.startswith("OLD_ID=")][0]
        print("进程A 发布旧套:", old_id)

        # 场景一：回读校验通过、替换中途被 SIGKILL -> 恢复新套
        r2 = run_worker(data_dir, "crash_mid_swap")
        assert r2.returncode == -signal.SIGKILL, (r2.returncode, r2.stderr)
        print("进程B 在替换中途被 SIGKILL（returncode=%d）" % r2.returncode)
        nv = Path(data_dir) / "new_volume"
        assert (nv / "pending_publish.json").is_file(), "待发布标记应已落盘"
        assert list(nv.glob(".backup-*")), "旧套应在备份目录"
        assert list((nv / "staging").glob("stg-*")), "新套应在暂存"

        r3 = run_worker(data_dir, "restart")
        assert r3.returncode == 0, r3.stderr
        recovered = [l for l in r3.stdout.splitlines() if l.startswith("RECOVERED_ID=")][0]
        new_id = recovered.split()[0].split("=", 1)[1]
        print("进程C 重启恢复:", recovered)
        assert new_id != old_id, "应恢复校验通过的新套，而不是旧套"

        r4 = run_worker(data_dir, "restart")
        assert r4.returncode == 0, r4.stderr
        again = [l for l in r4.stdout.splitlines() if l.startswith("RECOVERED_ID=")][0]
        assert again.split()[0].split("=", 1)[1] == new_id
        print("二次重启结果稳定:", again)

    with tempfile.TemporaryDirectory() as data_dir:
        # 场景二：回读校验前被 SIGKILL（暂存不完整、无 pending）-> 恢复旧套
        r1 = run_worker(data_dir, "publish_old")
        assert r1.returncode == 0, r1.stderr
        old_id = [l.split("=", 1)[1] for l in r1.stdout.splitlines()
                  if l.startswith("OLD_ID=")][0]
        r2 = run_worker(data_dir, "crash_before_verify")
        assert r2.returncode == -signal.SIGKILL, (r2.returncode, r2.stderr)
        print("\n进程B 在回读校验前被 SIGKILL（returncode=%d）" % r2.returncode)
        nv = Path(data_dir) / "new_volume"
        assert not (nv / "pending_publish.json").is_file(), "校验前不应有待发布标记"
        assert list((nv / "staging").glob("stg-*")), "应留下不完整暂存"

        r3 = run_worker(data_dir, "restart")
        assert r3.returncode == 0, r3.stderr
        recovered = [l for l in r3.stdout.splitlines() if l.startswith("RECOVERED_ID=")][0]
        got_id = recovered.split()[0].split("=", 1)[1]
        print("进程C 重启恢复:", recovered)
        assert got_id == old_id, "未通过全套校验必须恢复上次发布"
        assert not list((nv / "staging").glob("stg-*")), "不完整暂存应被清理"
    print("\n真实 SIGKILL 崩溃恢复演练通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
