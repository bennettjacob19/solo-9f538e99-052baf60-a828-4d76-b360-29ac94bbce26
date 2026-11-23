"""卷模拟与发布流程（含崩溃恢复）。

- 旧卷：目录模拟，可卸载 / 挂回。只有挂回同一卷（volume id 相同）才可读。
- 新卷：published/（当前发布的一套）+ staging/（本次暂存）。
  容量按字节限制，上一套发布与本次暂存都计入占用；记录分片文件占用峰值。
- 发布流程：读源 -> 校验（缺片 / 训练步）-> 重分片 -> 逐片暂存并累计容量
  -> 在指定挂载条件下回读校验 -> 写完整套标记与待发布标记 -> 原子替换发布
  -> 保存发布记录 -> 清理遗留。
  任何一步失败：清理暂存、保留上次发布结果、在历史里记录原因。

崩溃恢复（进程在任意时刻退出后重启）：
- 全套回读校验通过之前中断：暂存区没有待发布标记，恢复时丢弃暂存，
  从备份目录还原上次发布，页面发布编号与各 rank 下载仍指向上一套。
- 回读校验已通过（暂存目录带 manifest，且存在 pending_publish.json）：
  新套是完整套，恢复时优先发布新套；若新套意外不完整而备份完整，回退旧套，
  即“可恢复任一完整套”。
- state.json 已保存（页面已提示发布成功）后退出：published/ 与记录都指向新套，
  恢复只做遗留清理，保留新套。
- 恢复只读 manifest.json / pending_publish.json 等元数据，不读取 NPZ 正文、
  不复制分片；仅使用 rename 级移动与目录删除，每一步幂等可重入。
"""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

from reshard import (ARRAY_KEYS, OverCapacityError, ReshardError,
                     SimulatedWriteError, SourceUnreadableError, VerifyError,
                     load_source_checkpoint, reshard_arrays, split_sizes,
                     verify_shards, write_shard)

STAGING_PREFIX = "stg-"
BACKUP_PREFIX = ".backup-"
PENDING_NAME = "pending_publish.json"
MANIFEST_NAME = "manifest.json"
STATE_TMP_SUFFIX = ".tmp"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _dir_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _atomic_write_json(path: Path, payload: dict) -> None:
    """写 JSON 并原子替换，保证进程随时退出都不会留下半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + STATE_TMP_SUFFIX)
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


class OldVolume:
    """旧卷：mount.json 记录挂载状态；挂载的卷 id 与盘上 id 一致才可读。"""

    def __init__(self, root: Path):
        self.dir = Path(root) / "old_volume"
        self.mount_file = Path(root) / "mount.json"

    # ---- 挂载状态 ----
    def _read_mount(self) -> dict:
        if self.mount_file.is_file():
            try:
                return json.loads(self.mount_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {"mounted": False, "volume_id": None}

    def _write_mount(self, mounted: bool, volume_id: str | None) -> None:
        self.mount_file.parent.mkdir(parents=True, exist_ok=True)
        self.mount_file.write_text(
            json.dumps({"mounted": mounted, "volume_id": volume_id}, indent=2),
            encoding="utf-8",
        )

    def stored_id(self) -> str | None:
        f = self.dir / "volume.id"
        return f.read_text(encoding="utf-8").strip() if f.is_file() else None

    def status(self) -> dict:
        m = self._read_mount()
        sid = self.stored_id()
        return {
            "mounted": bool(m.get("mounted")),
            "volume_id": m.get("volume_id"),
            "stored_id": sid,
            "accessible": bool(m.get("mounted")) and m.get("volume_id") == sid,
        }

    def accessible(self) -> bool:
        return self.status()["accessible"]

    def mount(self, volume_id: str | None = None) -> None:
        """挂载；不传 id 表示挂回盘上那一块（同一卷）。"""
        self._write_mount(True, volume_id if volume_id is not None else self.stored_id())

    def unmount(self) -> None:
        self._write_mount(False, None)

    # ---- 内容 ----
    def checkpoints_dir(self) -> Path:
        return self.dir / "checkpoints"

    def list_checkpoints(self) -> list[dict] | None:
        """返回检查点摘要列表；旧卷不可读时返回 None。"""
        if not self.accessible():
            return None
        base = self.checkpoints_dir()
        out = []
        if base.is_dir():
            for d in sorted(base.iterdir()):
                mf = d / "manifest.json"
                if d.is_dir() and mf.is_file():
                    try:
                        m = json.loads(mf.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        m = {"ckpt_id": d.name, "error": "manifest 损坏"}
                    m.setdefault("ckpt_id", d.name)
                    out.append(m)
        return out


class NewVolume:
    """新卷：容量限制、published/staging 占用、峰值、历史与崩溃恢复。"""

    def __init__(self, root: Path):
        self.dir = Path(root) / "new_volume"
        self.published_root = self.dir / "published"
        self.staging_root = self.dir / "staging"
        self.state_file = self.dir / "state.json"
        self.pending_file = self.dir / PENDING_NAME
        for d in (self.published_root, self.staging_root):
            d.mkdir(parents=True, exist_ok=True)
        self._state = self._load()
        # 启动即恢复：先收敛上次未完成的发布，再清理遗留，全程不动 NPZ 正文。
        # 必须在全部路径属性赋值之后执行。
        self.recover()

    def _load(self) -> dict:
        data = _read_json(self.state_file)
        if data is not None:
            return data
        return {
            "capacity_bytes": 1 << 20,  # 默认 1 MiB
            "published": None,
            "peak_bytes": 0,
            "history": [],
        }

    def save(self) -> None:
        _atomic_write_json(self.state_file, self._state)

    # ---- 状态访问 ----
    @property
    def capacity(self) -> int:
        return int(self._state["capacity_bytes"])

    def set_capacity(self, capacity: int) -> None:
        self._state["capacity_bytes"] = int(capacity)

    @property
    def published(self) -> dict | None:
        return self._state["published"]

    @property
    def peak_bytes(self) -> int:
        return int(self._state["peak_bytes"])

    @property
    def history(self) -> list[dict]:
        return self._state["history"]

    def published_bytes(self) -> int:
        return _dir_bytes(self.published_root)

    def staging_bytes(self) -> int:
        return _dir_bytes(self.staging_root)

    def usage_bytes(self) -> int:
        """分片文件当前占用：上一套发布 + 本次暂存。

        遗留的暂存 / 备份在下次重分片开始前统一回收，不会长期计入。
        """
        return self.published_bytes() + self.staging_bytes()

    def record_peak(self, value: int) -> None:
        if value > self.peak_bytes:
            self._state["peak_bytes"] = int(value)

    def append_history(self, entry: dict) -> None:
        self._state["history"].append(entry)
        del self._state["history"][:-50]

    # ---- 完整套识别（只读 manifest 元数据，不读 NPZ 正文） ----
    @staticmethod
    def _set_manifest(set_dir: Path) -> dict | None:
        """目录是一套带 manifest 的完整分片套则返回记录，否则 None。

        只检查 manifest.json 与声明的 rank 分片是否齐全（文件名级），
        不打开、不复制任何 NPZ。
        """
        if not set_dir.is_dir():
            return None
        record = _read_json(set_dir / MANIFEST_NAME)
        if record is None or "id" not in record or "ranks" not in record:
            return None
        try:
            ranks = int(record["ranks"])
        except (TypeError, ValueError):
            return None
        if ranks < 1:
            return None
        for r in range(ranks):
            if not (set_dir / f"rank_{r}.npz").is_file():
                return None
        return record

    def _list_backup_dirs(self) -> list[Path]:
        if not self.dir.is_dir():
            return []
        return sorted(
            (p for p in self.dir.iterdir()
             if p.is_dir() and p.name.startswith(BACKUP_PREFIX)),
            key=lambda p: p.name,
        )

    @staticmethod
    def _backup_set_dir(backup: Path) -> Path | None:
        """备份目录是整个 published/ 移出来的，完整套位于其下唯一的 <id>/ 子目录。"""
        if not backup.is_dir():
            return None
        children = [p for p in backup.iterdir() if p.is_dir()]
        return children[0] if len(children) == 1 else None

    def _new_backup_dir(self) -> Path:
        return self.dir / f"{BACKUP_PREFIX}{uuid.uuid4().hex[:8]}"

    def _restore_backup(self) -> dict | None:
        """把最近一个完整备份还原为 published，返回其 manifest 记录。"""
        for backup in reversed(self._list_backup_dirs()):
            set_dir = self._backup_set_dir(backup)
            record = self._set_manifest(set_dir) if set_dir is not None else None
            if record is None:
                shutil.rmtree(backup, ignore_errors=True)
                continue
            if self.published_root.exists():
                shutil.rmtree(self.published_root, ignore_errors=True)
            shutil.move(str(backup), str(self.published_root))
            # 还原后该套即位于 published/<id>，与正常发布布局一致。
            return record
        return None

    # ---- 崩溃恢复 ----
    def recover(self) -> None:
        """进程启动时收敛上次发布的中间状态。幂等，可安全重复执行。"""
        pending = _read_json(self.pending_file)
        if pending is not None:
            self._converge_pending(pending)
        else:
            self._converge_without_pending()
        self.reclaim_leftovers()
        self.save()

    def _converge_pending(self, pending: dict) -> None:
        """回读校验已通过、待发布标记仍在：优先收敛到新套。

        新套完整 -> 发布新套（页面已提示成功与否都保留新套）；
        新套不完整而旧备份完整 -> 回退旧套（恢复任一完整套）。
        """
        target = self.published_root / str(pending.get("id", ""))
        new_record = self._set_manifest(target)
        if new_record is None:
            # 目标名之外，published 下可能已有落位的完整套（替换已完成、
            # 仅标记没来得及清），按盘上实际内容识别。
            if self.published_root.is_dir():
                for d in self.published_root.iterdir():
                    record = self._set_manifest(d)
                    if record is not None:
                        new_record = record
                        target = d
                        break
        if new_record is None:
            staging = self.staging_root / str(pending.get("staging", ""))
            new_record = self._set_manifest(staging)
            if new_record is not None:
                # 替换进行到一半：把旧套先挪到备份，再把新套落入 published。
                # 暂存目录本身是 stg-xxx（不带 id 层），落位目标须用发布 id 命名。
                target = self.published_root / str(new_record["id"])
                backup = self._new_backup_dir()
                had_old = self.published_root.exists() and any(
                    self.published_root.iterdir())
                if had_old:
                    shutil.move(str(self.published_root), str(backup))
                self.published_root.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(staging), str(target))
                except Exception:
                    if had_old and backup.exists():
                        shutil.rmtree(self.published_root, ignore_errors=True)
                        shutil.move(str(backup), str(self.published_root))
                    raise
        if new_record is None:
            # 新套已不完整：回退到完整的旧套；都没有则发布置空。
            new_record = self._restore_backup()
        self._state["published"] = new_record

    def _converge_without_pending(self) -> None:
        """没有待发布标记：新套尚未通过全套回读校验，恢复上次发布。"""
        # 只有“内含完整套”的备份才代表替换中途把 published 挪走了；
        # 空壳 / 损坏的备份与其它遗留目录一样直接清理，不动当前发布。
        restorable: tuple[Path, dict] | None = None
        for backup in self._list_backup_dirs():
            set_dir = self._backup_set_dir(backup)
            record = self._set_manifest(set_dir) if set_dir is not None else None
            if record is not None:
                restorable = (backup, record)
            else:
                shutil.rmtree(backup, ignore_errors=True)
        if restorable is not None:
            backup, record = restorable
            if self.published_root.exists():
                shutil.rmtree(self.published_root, ignore_errors=True)
            shutil.move(str(backup), str(self.published_root))
            self._state["published"] = record
            return
        # 没有可用备份：published 布局应当未受影响，按盘上完整套校正记录，
        # 保证页面发布编号与各 rank 下载始终指向同一套。
        if self.published_root.exists():
            complete = [
                d for d in self.published_root.iterdir()
                if d.is_dir() and self._set_manifest(d) is not None
            ]
            if complete:
                keep = max(complete, key=lambda d: d.name)
                for d in self.published_root.iterdir():
                    if d.is_dir() and d != keep:
                        shutil.rmtree(d, ignore_errors=True)
                self._state["published"] = self._set_manifest(keep)
                return
            # published 下没有任何完整套，记录与盘面对齐为空。
        self._state["published"] = None

    def reclaim_leftovers(self) -> None:
        """回收遗留占用：未完成的暂存、替换备份、残留临时文件与待发布标记。

        全部使用 ignore_errors 的删除，清理期间页面仍可打开、当前套仍可下载。
        下次重分片的容量判断在本方法之后进行，遗留不会再占容量。
        """
        if self.staging_root.is_dir():
            for d in self.staging_root.iterdir():
                if d.is_dir() and d.name.startswith(STAGING_PREFIX):
                    shutil.rmtree(d, ignore_errors=True)
        for backup in self._list_backup_dirs():
            shutil.rmtree(backup, ignore_errors=True)
        for tmp in self.dir.glob(f"*{STATE_TMP_SUFFIX}"):
            tmp.unlink(missing_ok=True)
        if self.pending_file.is_file():
            self.pending_file.unlink(missing_ok=True)

    # ---- 发布 ----
    def new_staging_dir(self) -> Path:
        d = self.staging_root / f"{STAGING_PREFIX}{uuid.uuid4().hex[:12]}"
        d.mkdir(parents=True, exist_ok=False)
        return d

    def publish(self, staging_dir: Path, record: dict) -> None:
        """用暂存目录整体替换当前发布。

        前置：全套回读校验已通过，暂存目录内已写好 manifest.json（完整套标记），
        且 pending_publish.json 已原子落盘（崩溃恢复据此识别“已通过校验的新套”）。
        旧套先挪为备份，新套落入后再删，移动中途失败可回滚到旧套。
        """
        target = self.published_root / record["id"]
        backup = self._new_backup_dir()
        had_old = self.published_root.exists() and any(self.published_root.iterdir())
        if had_old:
            shutil.move(str(self.published_root), str(backup))
        self.published_root.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(staging_dir), str(target))
        except Exception:
            if had_old and backup.exists():
                shutil.rmtree(self.published_root, ignore_errors=True)
                shutil.move(str(backup), str(self.published_root))
            raise
        # 新套已落位并自带 manifest；先收敛状态再清理备份，
        # 任一时刻崩溃，重启后 recover() 都能从 pending 标记恢复出新套。
        self._state["published"] = record
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        self.pending_file.unlink(missing_ok=True)


class ReshardService:
    """把 OldVolume / NewVolume 串起来的业务流程。"""

    def __init__(self, data_dir: Path):
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.old = OldVolume(data_dir)
        self.new = NewVolume(data_dir)
        self.lock = threading.Lock()

    # ---- 可读性 ----
    def published_readable(self) -> bool | None:
        """已发布结果当前是否可读；无发布返回 None。

        依赖旧卷的结果：旧卷卸载或挂了不同的卷 -> 不可读；挂回同一卷 -> 恢复。
        """
        pub = self.new.published
        if pub is None:
            return None
        if not pub.get("requires_old_volume"):
            return True
        st = self.old.status()
        return bool(st["mounted"] and st["volume_id"] == pub.get("old_volume_id"))

    # ---- 重分片主流程 ----
    def run(self, ckpt_id: str, n_new: int, capacity: int,
            can_mount_old: bool, simulate_write_fail: bool) -> dict:
        with self.lock:
            # 先释放上次中断留下的暂存 / 备份占用，再按原规则判断容量。
            self.new.reclaim_leftovers()
            op = {
                "time": _now(),
                "ckpt_id": ckpt_id,
                "n_new": n_new,
                "capacity": capacity,
                "can_mount_old": can_mount_old,
                "simulate_write_fail": simulate_write_fail,
            }
            staging_dir = self.new.new_staging_dir()
            try:
                if capacity < 0:
                    raise ReshardError("新卷容量必须 >= 0")
                self.new.set_capacity(capacity)
                op.update(self._do_run(ckpt_id, n_new, can_mount_old,
                                       simulate_write_fail, staging_dir))
                op["status"] = "published"
            except ReshardError as exc:
                shutil.rmtree(staging_dir, ignore_errors=True)
                op["status"] = "failed"
                op["reason"] = str(exc)
            except Exception as exc:  # noqa: BLE001 - 未预期错误同样保留上次发布
                shutil.rmtree(staging_dir, ignore_errors=True)
                op["status"] = "failed"
                op["reason"] = f"内部错误: {exc!r}"
            # 失败路径同样不允许残留暂存 / 待发布标记占用容量。
            self.new.reclaim_leftovers()
            self.new.append_history(op)
            self.new.save()
            return op

    def _do_run(self, ckpt_id: str, n_new: int, can_mount_old: bool,
                simulate_write_fail: bool, staging_dir: Path) -> dict:
        if n_new < 1:
            raise ReshardError("新 rank 数必须 >= 1")
        if not self.old.accessible():
            raise SourceUnreadableError("旧卷未挂载或挂载了不同的卷，源检查点不可读")

        # 1) 读源 + 校验（缺片 / 训练步不一致 / 段长度）
        src = load_source_checkpoint(self.old.checkpoints_dir() / ckpt_id)

        # 2) 三个数组按同一全局下标连续等分，余数给低 rank
        shards = {k: reshard_arrays(src.arrays[k], n_new) for k in ARRAY_KEYS}
        sizes = split_sizes(src.total, n_new)

        # 3) 逐片暂存；上一套发布 + 本次暂存一起计入容量
        published_before = self.new.published_bytes()
        staged = 0
        shard_files: list[Path] = []
        fail_at = n_new // 2 if simulate_write_fail else -1
        for r in range(n_new):
            if r == fail_at:
                raise SimulatedWriteError(f"模拟写失败: 写入 rank_{r}.npz 时出错")
            path = staging_dir / f"rank_{r}.npz"
            staged += write_shard(path, {k: shards[k][r] for k in ARRAY_KEYS}, src.step)
            shard_files.append(path)
            if published_before + staged > self.new.capacity:
                raise OverCapacityError(
                    f"新卷容量超限: 上一套发布 {published_before} B + "
                    f"本次暂存 {staged} B > 容量 {self.new.capacity} B"
                )

        # 4) 在指定挂载条件下回读校验全套分片，可读才发布
        old_st = self.old.status()
        if can_mount_old and not old_st["accessible"]:
            raise VerifyError("声明续训时可挂旧卷，但旧卷当前不可读，"
                              "无法在指定挂载条件下验证全套分片")
        verify_shards(shard_files, sizes, src.step, src)

        # 5) 构造发布记录；manifest 先写进暂存目录（完整套标记），
        #    再原子落盘待发布标记 —— 此后崩溃重启都会优先恢复这套新分片。
        op_peak = published_before + staged
        self.new.record_peak(op_peak)
        pub_id = f"pub-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        record = {
            "id": pub_id,
            "source_ckpt": src.ckpt_id,
            "step": src.step,
            "ranks": n_new,
            "total_elems": src.total,
            "dtype": src.dtype,
            "shard_sizes": sizes,
            "bytes": staged,
            "requires_old_volume": bool(can_mount_old),
            "old_volume_id": old_st["volume_id"] if can_mount_old else None,
            "created_at": _now(),
        }
        _atomic_write_json(staging_dir / MANIFEST_NAME, record)
        _atomic_write_json(self.new.pending_file, {
            "id": pub_id,
            "staging": staging_dir.name,
            "created_at": record["created_at"],
        })

        # 6) 原子替换发布；publish 内部收敛 published 记录并清理备份 / 标记
        self.new.publish(staging_dir, record)
        self.new.record_peak(self.new.published_bytes())
        return {
            "publish_id": pub_id,
            "step": src.step,
            "shard_sizes": sizes,
            "staged_bytes": staged,
            "prev_published_bytes": published_before,
            "op_peak": op_peak,
        }
