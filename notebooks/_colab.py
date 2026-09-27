"""Colab session glue for AlignForge.

Environment plumbing only: Drive layout, env overrides, durable state,
registry snapshots, a streaming CLI runner, artifact discovery, and a
pre-GPU code audit. No training logic lives here. Every ML step goes
through the `alignforge` CLI in a child process, so the notebook kernel
never imports torch and never holds CUDA memory.

Standard library only, and never imports google.colab, so it is
unit-testable in CI (tests/unit/test_colab_helpers.py).
"""

from __future__ import annotations

import codecs
import json
import os
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Root-relative dirs with no env override in core/paths.py -> mirrored to Drive.
MIRRORED_DIRS: tuple[str, ...] = ("logs", "reports", "evals")


@dataclass(frozen=True)
class Layout:
    """Where everything lives in a Colab session.

    Durable (Drive): datasets, run artifacts, exported models, state, stage logs.
    Fast (local disk): the live SQLite registry and the HuggingFace cache.
    """

    repo: Path
    drive_root: Path
    local_root: Path = Path("/content/af_local")

    # -- durable (Google Drive) ------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.drive_root / "data"

    @property
    def artifacts_dir(self) -> Path:
        return self.drive_root / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.drive_root / "models"

    @property
    def mirror_dir(self) -> Path:
        return self.drive_root / "mirror"

    @property
    def registry_snapshot(self) -> Path:
        return self.drive_root / "registry" / "alignforge.db"

    @property
    def state_file(self) -> Path:
        return self.drive_root / "state.json"

    @property
    def stage_logs_dir(self) -> Path:
        return self.drive_root / "stage_logs"

    # -- fast (local VM disk, lost on disconnect) --------------------------
    @property
    def registry_db(self) -> Path:
        return self.local_root / "alignforge.db"

    @property
    def hf_home(self) -> Path:
        return self.local_root / "hf_cache"

    def ensure(self) -> None:
        """Create every directory in the layout (idempotent)."""
        for d in (
            self.data_dir,
            self.artifacts_dir,
            self.models_dir,
            self.mirror_dir,
            self.registry_snapshot.parent,
            self.stage_logs_dir,
            self.local_root,
            self.hf_home,
        ):
            d.mkdir(parents=True, exist_ok=True)


def export_env(layout: Layout) -> dict[str, str]:
    """Point AlignForge's path overrides at the layout; child processes inherit them."""
    env = {
        "ALIGNFORGE_DATA_DIR": str(layout.data_dir),
        "ALIGNFORGE_ARTIFACTS_DIR": str(layout.artifacts_dir),
        "ALIGNFORGE_MODELS_DIR": str(layout.models_dir),
        "ALIGNFORGE_REGISTRY_DB": str(layout.registry_db),
        "HF_HOME": str(layout.hf_home),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONUNBUFFERED": "1",
    }
    os.environ.update(env)
    return env


class State:
    """Tiny JSON key-value store on Drive: dataset hashes, run ids, stage flags.

    This is what makes every notebook stage idempotent: a stage checks
    state.json, skips if its output is recorded, and records it when done.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        return data

    def get(self, key: str, default: Any = None) -> Any:
        return self.load().get(key, default)

    def set(self, **values: Any) -> None:
        data = self.load()
        data.update(values)
        data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)  # write-then-rename: never a half-written state file

    def require(self, key: str) -> Any:
        value = self.get(key)
        if value in (None, ""):
            raise RuntimeError(f"state.json has no {key!r}. Run the stage that produces it first.")
        return value


def _sqlite_copy(src: Path, dst: Path) -> None:
    """Consistent copy of a live SQLite DB (safe while a writer uses WAL mode)."""
    dst.unlink(missing_ok=True)
    source = sqlite3.connect(src)
    target = sqlite3.connect(dst)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


def snapshot_registry(layout: Layout) -> Path | None:
    """Back up the local registry to Drive, keeping one previous generation."""
    if not layout.registry_db.is_file():
        return None
    tmp = layout.local_root / "alignforge.snapshot.db"
    _sqlite_copy(layout.registry_db, tmp)  # backup API on local disk, not on FUSE
    dst = layout.registry_snapshot
    if dst.is_file():
        shutil.copy2(dst, dst.with_name("alignforge.prev.db"))
    shutil.copy2(tmp, dst)
    return dst


def restore_registry(layout: Layout) -> str:
    """Seed the local registry from the Drive snapshot at session start."""
    if layout.registry_db.is_file():
        return "local registry already present (same VM)"
    if layout.registry_snapshot.is_file():
        shutil.copy2(layout.registry_snapshot, layout.registry_db)
        return f"restored from {layout.registry_snapshot}"
    return "no snapshot yet (fresh project)"


def copy_tree(src: Path, dst: Path, *, overwrite: bool) -> int:
    """Copy files src -> dst. Skips unchanged files; returns number copied."""
    if not src.is_dir():
        return 0
    copied = 0
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        target = dst / f.relative_to(src)
        if target.exists():
            if not overwrite:
                continue
            s, t = f.stat(), target.stat()
            if s.st_size == t.st_size and t.st_mtime >= s.st_mtime:
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        copied += 1
    return copied


def restore_mirror(layout: Layout) -> dict[str, int]:
    """Drive -> repo, never overwriting files git already provided."""
    return {
        name: copy_tree(layout.mirror_dir / name, layout.repo / name, overwrite=False)
        for name in MIRRORED_DIRS
    }


def persist(layout: Layout) -> dict[str, Any]:
    """Repo -> Drive for mirrored dirs, plus a registry snapshot. Call after every stage."""
    mirrored = {
        name: copy_tree(layout.repo / name, layout.mirror_dir / name, overwrite=True)
        for name in MIRRORED_DIRS
    }
    snap = snapshot_registry(layout)
    return {"mirrored": mirrored, "registry_snapshot": str(snap) if snap else None}


def run(
    cmd: list[str],
    *,
    stage: str,
    layout: Layout,
    extra_env: dict[str, str] | None = None,
    check: bool = True,
    heartbeat: Callable[[], Any] | None = None,
    heartbeat_every: float = 600.0,
) -> str:
    """Run a command in a child process: stream live, tee to Drive, return output.

    Streams raw bytes (so tqdm's carriage returns render), decodes with an
    incremental UTF-8 decoder (so multi-byte chars split across reads survive),
    and on a cell interrupt forwards SIGINT so the child releases the GPU.
    """
    env = {**os.environ, **(extra_env or {})}
    log_path = layout.stage_logs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{stage}.log"
    print(f"$ {shlex.join(cmd)}\n  [log -> {log_path}]", flush=True)

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    chunks: list[str] = []
    last_flush = last_beat = time.monotonic()
    proc = subprocess.Popen(
        cmd, cwd=layout.repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    try:
        with (log_path).open("w", encoding="utf-8") as log_f:
            while True:
                buf = os.read(fd, 4096)
                if not buf:
                    break
                text = decoder.decode(buf)
                sys.stdout.write(text)
                sys.stdout.flush()
                log_f.write(text)
                chunks.append(text)
                now = time.monotonic()
                if now - last_flush > 30:
                    log_f.flush()
                    last_flush = now
                if heartbeat is not None and now - last_beat > heartbeat_every:
                    try:
                        heartbeat()
                    except Exception as exc:  # a failed backup must never kill training
                        print(f"\n[heartbeat failed: {exc}]", flush=True)
                    last_beat = now
            log_f.write(decoder.decode(b"", final=True))
    except KeyboardInterrupt:
        print(f"\n[{stage}] interrupted: stopping child so it releases the GPU...", flush=True)
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        rest = proc.stdout.read()  # show the child's cleanup output (e.g. fail_run)
        if rest:
            sys.stdout.write(decoder.decode(rest, final=True))
        raise
    rc = proc.wait()
    if check and rc != 0:
        raise RuntimeError(f"[{stage}] exited with code {rc}. Full log: {log_path}")
    return "".join(chunks)


def has_flag(cmd: list[str], flag: str, *, layout: Layout) -> bool:
    """True if `cmd --help` advertises `flag` (lets the notebook adapt to the CLI)."""
    env = {**os.environ, "COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"}
    res = subprocess.run([*cmd, "--help"], cwd=layout.repo, env=env, capture_output=True, text=True)
    return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", res.stdout) is not None


def gpu_status() -> str:
    """One-line GPU summary from nvidia-smi (no torch import, no CUDA context)."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "no GPU visible"
    return res.stdout.strip()


def smoke_env(layout: Layout, name: str) -> dict[str, str]:
    """Hermetic env for a smoke run: throwaway artifacts dir + a copy of the registry.

    Smoke runs then never pollute the durable registry or Drive, but can still
    resolve datasets and parent runs recorded in the real registry.
    """
    root = layout.local_root / "smoke" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    db = root / "alignforge.db"
    if layout.registry_db.is_file():
        _sqlite_copy(layout.registry_db, db)
    return {
        "ALIGNFORGE_ARTIFACTS_DIR": str(root / "artifacts"),
        "ALIGNFORGE_MODELS_DIR": str(root / "models"),
        "ALIGNFORGE_REGISTRY_DB": str(db),
    }


def mark_failed(run_id: str, reason: str, *, layout: Layout) -> None:
    """Best-effort: flip an interrupted run's registry row to failed."""
    code = (
        "import sys\n"
        "from alignforge.core.registry import get_registry\n"
        "get_registry().fail_run(sys.argv[1], error=sys.argv[2])\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code, run_id, reason],
        cwd=layout.repo,
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        print(f"registry: marked {run_id} as failed ({reason})")
    else:
        # Expected when the row never reached a snapshot before the disconnect.
        print(f"registry: could not mark {run_id} failed (probably not in the snapshot)")


_HEX = re.compile(r"\b[0-9a-f]{8,64}\b")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_RUN_ID = re.compile(r"Run ID:\s*(\S+)")


def list_manifests(data_dir: Path) -> dict[Path, float]:
    """Every dataset manifest under data_dir, with its mtime."""
    return {p: p.stat().st_mtime for p in data_dir.rglob("manifest.json")}


def _manifest_hash(path: Path) -> str | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("content_hash")
    except (OSError, ValueError):
        return None
    return str(value) if value else None


def detect_new_dataset_hash(before: dict[Path, float], data_dir: Path, output: str) -> str:
    """Content hash of the dataset a `data build` just produced.

    Primary: the manifest that appeared/changed during the build.
    Fallback: the last hex token on a hash line of the CLI output (covers a
    rebuild that produced an identical, already-existing artifact).
    """
    after = list_manifests(data_dir)
    changed = [p for p, mtime in after.items() if before.get(p) != mtime]
    hashes = sorted({h for p in changed if (h := _manifest_hash(p))})
    if len(hashes) == 1:
        return hashes[0]
    for line in reversed(_ANSI.sub("", output).splitlines()):
        low = line.lower()
        if "hash" in low and "config" not in low:
            found = _HEX.findall(low)
            if found:
                return str(found[-1])
    raise RuntimeError(
        "Could not detect the dataset hash. Read it from the output above and "
        "record it with S.set(sft_hash='...') or S.set(pref_hash='...')."
    )


def parse_run_id(output: str) -> str | None:
    """Run id from the CLI's 'Run ID: <id>' line (last occurrence wins)."""
    found = _RUN_ID.findall(_ANSI.sub("", output))
    return str(found[-1]) if found else None


@dataclass(frozen=True)
class RunDir:
    run_id: str
    path: Path
    complete: bool
    latest_checkpoint: Path | None
    mtime: float


def _ckpt_step(p: Path) -> int:
    tail = p.name.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def find_runs(artifacts_dir: Path, kind: str) -> list[RunDir]:
    """Run dirs `<kind>-*`, newest first.

    complete  = adapter/adapter_config.json exists (written only after train()).
    checkpoint = newest checkpoint-N holding trainer_state.json, which Trainer
                 writes last, so a checkpoint torn by a disconnect is skipped.
    """
    runs: list[RunDir] = []
    for d in artifacts_dir.glob(f"{kind}-*"):
        if not d.is_dir():
            continue
        ckpt_root = d / "checkpoints"
        ckpts = sorted(
            (c for c in ckpt_root.glob("checkpoint-*") if (c / "trainer_state.json").is_file()),
            key=_ckpt_step,
        )
        runs.append(
            RunDir(
                run_id=d.name,
                path=d,
                complete=(d / "adapter" / "adapter_config.json").is_file(),
                latest_checkpoint=ckpts[-1] if ckpts else None,
                mtime=d.stat().st_mtime,
            )
        )
    return sorted(runs, key=lambda r: r.mtime, reverse=True)


def latest_complete(artifacts_dir: Path, kind: str) -> RunDir | None:
    return next((r for r in find_runs(artifacts_dir, kind) if r.complete), None)


def latest_resumable(artifacts_dir: Path, kind: str) -> RunDir | None:
    return next(
        (r for r in find_runs(artifacts_dir, kind) if not r.complete and r.latest_checkpoint),
        None,
    )


def log_records(
    logs_dir: Path,
    *,
    event: str | None = None,
    key: str | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Structured JSON log lines filtered by event name, field presence, and run_id."""
    rows: list[dict[str, Any]] = []
    for log_file in sorted(logs_dir.glob("*.log")):
        with (log_file).open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if (event and event not in line) or (key and key not in line):
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if event and rec.get("event") != event:
                    continue
                if key and key not in rec:
                    continue
                if run_id and rec.get("run_id") != run_id:
                    continue
                rows.append(rec)
    return rows


def status(layout: Layout, state: State) -> None:
    """Session dashboard: GPU, recorded state, and run dirs on Drive."""
    print(f"GPU    : {gpu_status()}")
    print(f"state  : {layout.state_file}")
    for k, v in sorted(state.load().items()):
        print(f"  {k:24s} {v}")
    for kind in ("sft", "dpo"):
        for r in find_runs(layout.artifacts_dir, kind)[:5]:
            if r.complete:
                tag = "complete"
            elif r.latest_checkpoint:
                tag = f"resumable @ {r.latest_checkpoint.name}"
            else:
                tag = "incomplete, no usable checkpoint"
            print(f"  {r.run_id:34s} {tag}")


@dataclass(frozen=True)
class Finding:
    level: str  # "FIX" blocks the notebook; "WARN" is informational
    file: str
    message: str


def audit(repo: Path) -> list[Finding]:
    """Static checks for runtime bugs that only surface minutes into a GPU run."""
    findings: list[Finding] = []
    train = repo / "src" / "alignforge" / "train"
    sources = {p: p.read_text(encoding="utf-8") for p in sorted(train.glob("*.py"))}

    for path, src in sources.items():
        rel = str(path.relative_to(repo))
        if re.search(r"\.total_mem\b", src):
            findings.append(
                Finding(
                    "FIX",
                    rel,
                    "`.total_mem` should be `.total_memory` (AttributeError at the first VRAM log).",
                )
            )
        if re.search(r"(?<![\w.])Path\(", src) and not re.search(
            r"^\s*from pathlib import[^\n]*\bPath\b", src, re.M
        ):
            findings.append(
                Finding(
                    "FIX",
                    rel,
                    "`Path(` is used but `Path` is never imported (NameError).",
                )
            )

    cb = sources.get(train / "callbacks.py", "")
    if "def wrap_callbacks" in cb:
        callers = [
            p for p, s in sources.items() if p.name != "callbacks.py" and "wrap_callbacks(" in s
        ]
        if not callers:
            findings.append(
                Finding(
                    "FIX",
                    "src/alignforge/train/",
                    "Plain-class callbacks never go through wrap_callbacks(); Trainer calls "
                    "on_init_end() on them and raises AttributeError before step 1.",
                )
            )
        for p, s in sources.items():
            if p in callers or p.name == "callbacks.py":
                continue
            if re.search(r"callbacks\s*=\s*callbacks\b", s):
                findings.append(
                    Finding(
                        "WARN",
                        str(p.relative_to(repo)),
                        "Passes `callbacks` straight to a Trainer; confirm they were wrapped upstream.",
                    )
                )

    cli = repo / "src" / "alignforge" / "cli.py"
    if cli.is_file() and re.search(
        r"cfg\s*=\s*_lc\(\s*component_path\s*=\s*model_config", cli.read_text(encoding="utf-8")
    ):
        findings.append(
            Finding(
                "WARN",
                "src/alignforge/cli.py",
                "--model-config reloads the config and discards --config; check epochs/LR "
                "in the dry-run table match configs/train/*.yaml.",
            )
        )
    return findings


PKGS = ("torch", "transformers", "trl", "peft", "accelerate", "bitsandbytes", "datasets")


def preflight(require_gpu: bool = True) -> int:
    """Import-level checks for the exact APIs the training code calls. Runs in a child."""
    import importlib.metadata as md
    import inspect

    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'OK  ' if passed else 'FAIL'}] {label}  {detail}".rstrip())

    print("Packages:")
    for pkg in PKGS:
        try:
            print(f"  {pkg:14s} {md.version(pkg)}")
        except md.PackageNotFoundError:
            check(f"{pkg} installed", False)

    print("Runtime:")
    import torch

    cuda = torch.cuda.is_available()
    if require_gpu or cuda:
        check("CUDA GPU visible", cuda, torch.cuda.get_device_name(0) if cuda else "")
    if cuda:
        print(f"  bf16 supported: {torch.cuda.is_bf16_supported()}  (T4 -> False, fp16 used)")
    try:
        import bitsandbytes  # noqa: F401

        check("bitsandbytes imports", True)
    except Exception as exc:
        check("bitsandbytes imports", False, repr(exc))

    print("APIs used by src/alignforge/train:")
    import trl
    from transformers import TrainingArguments

    ta = inspect.signature(TrainingArguments.__init__).parameters
    check("TrainingArguments(evaluation_strategy=...)", "evaluation_strategy" in ta)
    check("trl.DataCollatorForCompletionOnlyLM", hasattr(trl, "DataCollatorForCompletionOnlyLM"))
    sft = inspect.signature(trl.SFTTrainer.__init__).parameters
    missing = [k for k in ("tokenizer", "dataset_text_field", "max_seq_length") if k not in sft]
    check(
        "SFTTrainer(tokenizer=, dataset_text_field=, max_seq_length=)",
        not missing,
        f"missing: {missing}" if missing else "",
    )
    check("trl.DPOTrainer", hasattr(trl, "DPOTrainer"))
    print("PREFLIGHT", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def _main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "preflight":
        return preflight(require_gpu="--cpu-ok" not in argv)
    if cmd == "audit":
        repo = Path(argv[2]) if len(argv) > 2 else Path.cwd()
        findings = audit(repo)
        for f in findings:
            print(f"  [{f.level:4s}] {f.file}: {f.message}")
        blocking = sum(f.level == "FIX" for f in findings)
        print(f"AUDIT: {blocking} blocking, {len(findings) - blocking} warnings")
        return 1 if blocking else 0
    print("usage: python notebooks/_colab.py {preflight [--cpu-ok] | audit [REPO]}")
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
