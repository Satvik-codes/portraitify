"""Single-worker job engine with state machine, persistence, and recovery."""
import json
import logging
import queue
import threading
import time
import uuid
from pathlib import Path

from config import config

log = logging.getLogger("h2v.jobs")

TERMINAL = ("done", "failed")


class Job:
    __slots__ = ("id", "status", "stage", "progress", "params",
                 "original_name", "width_in", "height_in",
                 "error", "created_at", "meta", "_lock")

    def __init__(self, job_id, params, original_name="", width_in=0, height_in=0):
        self.id = job_id
        self.status = "queued"
        self.stage = None
        self.progress = 0
        self.params = params
        self.original_name = original_name
        self.width_in, self.height_in = width_in, height_in
        self.error = None
        self.created_at = time.time()
        self.meta = {}
        self._lock = threading.Lock()

    def snapshot(self):
        with self._lock:
            return {
                "job_id": self.id, "status": self.status, "stage": self.stage,
                "progress": self.progress, "params": self.params,
                "error": self.error, "meta": self.meta,
                "result_url": f"/api/results/{self.id}" if self.status == "done" else None,
                "original_url": f"/api/uploads/{self.id}",
            }

    # mutations -------------------------------------------------------------
    def set(self, status=None, stage=None, progress=None, error=None):
        with self._lock:
            if status is not None:
                self.status = status
            if stage is not None:
                self.stage = stage
            if progress is not None:
                self.progress = max(self.progress, int(progress))
            if error is not None:
                self.error = str(error)[:400]

    def to_line(self):
        s = self.snapshot()
        s["original_name"] = self.original_name
        s["width_in"], s["height_in"] = self.width_in, self.height_in
        s["created_at"] = self.created_at
        s["meta"] = self.meta
        return json.dumps(s)


class JobManager:
    QUEUE_MAX = 20

    def __init__(self):
        self._jobs = {}                      # id -> Job (all live records)
        self._q = queue.Queue(maxsize=self.QUEUE_MAX)
        self._single = threading.Semaphore(1)   # one pipeline at a time
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, name="h2v-worker",
                                        daemon=True)
        self._jsonl = config.LOGS_DIR / "jobs.jsonl"
        self._recover_and_purge()

    # public ----------------------------------------------------------------
    def submit(self, upload_path: Path, params, original_name="",
               width_in=0, height_in=0) -> Job:
        job_id = uuid.uuid4().hex
        try:
            upload_path.rename(config.UPLOADS_DIR / f"{job_id}.bin")
        except OSError:
            pass  # route layer already stored atomically under same name
        job = Job(job_id, params, original_name, width_in, height_in)
        self._jobs[job_id] = job
        self._q.put_nowait(job)              # raises queue.Full -> caller maps 503
        self._persist(job)
        log.info("submitted %s %s", job_id, params)
        return job

    def get(self, job_id) -> Job | None:
        return self._jobs.get(job_id)

    def delete(self, job_id) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        for p in (config.UPLOADS_DIR / f"{job_id}.bin",
                  config.RESULTS_DIR / f"{job_id}.png"):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        if job.status == "queued":
            job.set(status="failed", error="deleted by user")
            self._persist(job)
        self._jobs.pop(job_id, None)
        return True

    def start(self):
        self._worker.start()

    def stop(self):
        self._stop.set()
        self._q.put(None)

    # internals ---------------------------------------------------------------
    def _loop(self):
        while not self._stop.is_set():
            job = self._q.get()
            if job is None:
                break
            if job.status != "queued":
                continue                     # deleted while queued
            with self._single:
                self._execute(job)

    def _execute(self, job: Job):
        from pipeline import run_smart

        def cb(stage, pct):
            job.set(stage=stage, progress=pct)
            self._persist(job)

        tmp_result = config.RESULTS_DIR / f"{job.id}.png"
        try:
            tmp_result.unlink(missing_ok=True)
            job.set(status="running")
            self._persist(job)
            img = _load_upload(config.UPLOADS_DIR / f"{job.id}.bin")
            meta = run_smart(img, ratio=job.params["ratio"],
                             tier=job.params["tier"],
                             align=job.params["align"],
                             job_id=job.id, cb=cb)
            if not tmp_result.exists():
                raise IOError("pipeline reported success but result file missing")
            job.meta = {k: v for k, v in meta.items() if k != "result_path"}
            job.set(status="done", stage=None, progress=100)
            self._persist(job)
            log.info("job %s done %s", job.id, job.meta.get("timings"))
        except Exception as e:
            tmp_result.unlink(missing_ok=True)
            msg = _user_message(e)
            job.set(status="failed", error=msg)
            self._persist(job)
            log.exception("job %s failed: %s", job.id, e)
        finally:
            from pipeline.vrac import release_all
            release_all()

    def _persist(self, job: Job):
        try:
            with open(self._jsonl, "a", encoding="utf-8") as f:
                f.write(job.to_line() + "\n")
        except OSError:
            log.exception("persist failed for %s", job.id)

    def _recover_and_purge(self):
        """Boot hygiene: stale rows -> failed; TTL + count purge on results."""
        if self._jsonl.exists():
            seen_ids = []
            try:
                for line in self._jsonl.read_text(encoding="utf-8").splitlines()[-5000:]:
                    try:
                        row = json.loads(line)
                        seen_ids.append((row.get("job_id"), row.get("status")))
                    except json.JSONDecodeError:
                        continue
            except OSError:
                pass
            latest = {}
            for jid, st in seen_ids:
                latest[jid] = st
            for jid, st in latest.items():
                if st in ("queued", "running"):
                    (config.RESULTS_DIR / f"{jid}.png").unlink(missing_ok=True)
                    row = {"job_id": jid, "status": "failed", "stage": None,
                           "progress": 0, "error": "server restarted mid-job",
                           "params": {}, "result_url": None}
                    try:
                        with open(self._jsonl, "a", encoding="utf-8") as f:
                            f.write(json.dumps(row) + "\n")
                    except OSError:
                        pass

        now = time.time()
        files = sorted(config.RESULTS_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime)
        keep = files[-config.RESULT_KEEP_COUNT:]
        for p in files[:len(files) - config.RESULT_KEEP_COUNT]:
            p.unlink(missing_ok=True)
        for p in keep:
            age_h = (now - p.stat().st_mtime) / 3600
            if age_h > config.RESULT_TTL_HOURS:
                p.unlink(missing_ok=True)
        for p in config.UPLOADS_DIR.glob("*.bin"):
            if (now - p.stat().st_mtime) / 3600 > config.RESULT_TTL_HOURS:
                p.unlink(missing_ok=True)


def _load_upload(path: Path):
    """Decode stored upload bytes to RGB ndarray. Worker-side only."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        return np.array(im.convert("RGB"))


def _user_message(e: Exception) -> str:
    name = type(e).__name__
    if name == "QualityNotInstalledError":
        return str(e)
    text = str(e).lower()
    if "out of memory" in text:
        return ("GPU ran out of memory during this conversion. "
                "Try the Fast tier or a smaller image.")
    return f"Conversion failed ({name}). See logs\\app.log for details."


_manager = None


def get_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
        _manager.start()
    return _manager
