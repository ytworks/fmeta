"""Runs child processes on a fixed set of devices (FM-D14).

A job is one or more tasks (child processes) that run at the same time: a replica is a job
with one task, a replica with shared-bias walkers a job with one task per walker. One loop in
the calling thread hands out the devices, so the same jobs on the same devices are always
assigned the same way. Every decision is appended to ``schedule.jsonl``, and
:func:`verify_schedule` checks such a record against the rules:

- Every entry of the device list is a place for one job at a time: a device named twice
  carries two jobs side by side (``run.per_gpu``). A job takes as many places as it asks
  for (all of them, if there are fewer) and spreads its tasks over their devices.
- A job starts when the jobs it comes after have ended successfully and its devices are free.
  Jobs start in the order of the list: a job that waits for devices holds back the jobs behind
  it, a job that waits for another job does not. Free places are handed out in list order.
- A task that fails stops the other tasks of its job; the other jobs go on. A job that comes
  after a job that did not end successfully is not started.
- After a stop request (SIGINT, SIGTERM) nothing is started, every child is terminated with
  its process group, and the call returns when the last one has ended.
"""

from __future__ import annotations

import fcntl
import json
import os
import queue
import signal
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

DONE, FAILED, STOPPED, SKIPPED = "done", "failed", "stopped", "skipped"


@dataclass(frozen=True)
class Task:
    """One child process. The launcher adds ``--device N`` for the device it is given."""

    name: str
    command: Sequence[str]
    log: Path


@dataclass(frozen=True)
class Job:
    key: str
    tasks: Sequence[Task]
    devices: int = 1                                    # devices the job occupies
    after: Sequence[str] = ()                           # keys of earlier jobs that must have ended successfully
    before_start: Optional[Callable[[], None]] = None   # called right before the tasks are launched


class _Child:
    """A child process in its own process group."""

    def __init__(self, process: subprocess.Popen):
        self.process, self.pid = process, process.pid

    def wait(self) -> int:
        return self.process.wait()

    def _signal(self, signum: int) -> None:
        try:
            os.killpg(self.pid, signum)             # the child leads its group: this reaches what it started
        except ProcessLookupError:
            pass

    def terminate(self) -> None:
        self._signal(signal.SIGTERM)

    def kill(self) -> None:
        self._signal(signal.SIGKILL)


def launch_subprocess(task: Task, device) -> _Child:
    """Start ``task`` as a child process; its output is appended to ``task.log``."""
    command = [str(part) for part in task.command]
    if device is not None:
        command += ["--device", str(device)]
    Path(task.log).parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    environment.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")      # device numbers as nvidia-smi shows them
    # One math-library thread per replica: the work is on the GPU. With the default (a thread per
    # CPU core in every process) 41 replicas on 128 cores gave a load average above 1200, and the
    # limit made them 3 to 30% faster (measured 2026-11-16). A value the caller sets is kept.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment.setdefault(variable, "1")
    with open(task.log, "a") as logfile:
        return _Child(subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=logfile,
                                       stderr=subprocess.STDOUT, env=environment,
                                       start_new_session=True))


class _Running:
    def __init__(self, job: Job, places: list, devices: list):
        self.job, self.places, self.devices = job, places, devices
        self.handles: dict = {}
        self.codes: dict = {}
        self.cause: Optional[str] = None            # why its tasks were told to stop, if they were

    def stop(self, cause: str, hard: bool = False) -> None:
        self.cause = self.cause or cause
        for name, handle in self.handles.items():
            if name not in self.codes:
                handle.kill() if hard else handle.terminate()


class Scheduler:
    """See the module text. ``launcher(task, device)`` returns a handle with ``pid``, ``wait()``,
    ``terminate()`` and ``kill()``; ``record(event, **fields)`` receives every decision."""

    def __init__(self, jobs: Sequence[Job], devices: Sequence, launcher=None, record=None,
                 grace: float = 5.0):
        self.jobs, self.devices = list(jobs), list(devices)
        self.launcher = launcher or launch_subprocess
        self.record = record or (lambda event, **fields: None)
        self.grace = grace                          # seconds a terminated child gets before it is killed
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        if not self.devices:
            raise ValueError("there must be at least one device")
        seen: set = set()
        for job in self.jobs:
            if job.key in seen or not job.tasks or job.devices < 1:
                raise ValueError(f"job {job.key}: keys are unique, a job has tasks and takes a device")
            if any(key not in seen for key in job.after):
                raise ValueError(f"job {job.key} comes after {list(job.after)}: those must be earlier in the list")
            seen.add(job.key)

    def request_stop(self, reason: str = "stop requested") -> None:
        """Safe to call from a signal handler and from other threads."""
        self._inbox.put(("stop", reason))

    def _wait(self, key: str, name: str, handle) -> None:
        self._inbox.put(("exit", key, name, handle.wait()))

    def _start(self, job: Job, places: list) -> Optional[_Running]:
        mine = [self.devices[place] for place in places]
        running = _Running(job, places, mine)
        try:
            if job.before_start is not None:
                job.before_start()
            for index, task in enumerate(job.tasks):
                running.handles[task.name] = self.launcher(task, mine[index % len(mine)])
        except Exception as error:                  # nothing of this job stays behind
            for handle in running.handles.values():
                handle.kill()
                handle.wait()
            self.record("skip", job=job.key, reason=f"could not start: {error}")
            return None
        self.record("start", job=job.key, devices=mine,
                    tasks={task.name: mine[index % len(mine)] for index, task in enumerate(job.tasks)},
                    pids={name: getattr(handle, "pid", None) for name, handle in running.handles.items()})
        for name, handle in running.handles.items():
            threading.Thread(target=self._wait, args=(job.key, name, handle), daemon=True).start()
        return running

    def run(self) -> dict:
        """Run every job; returns ``{key: "done" | "failed" | "stopped" | "skipped"}``."""
        need = {job.key: min(job.devices, len(self.devices)) for job in self.jobs}
        status: dict = {}
        waiting, running = list(self.jobs), {}
        free = set(range(len(self.devices)))        # places: positions in the device list
        stopping: Optional[str] = None
        self.record("begin", devices=self.devices,
                    jobs=[dict(key=job.key, tasks=[task.name for task in job.tasks], devices=need[job.key],
                               after=list(job.after)) for job in self.jobs])
        while True:
            still, held_back = [], False
            for job in waiting:
                broken = [key for key in job.after if status.get(key, DONE) != DONE]
                if stopping is not None or broken:
                    status[job.key] = SKIPPED
                    self.record("skip", job=job.key,
                                reason=stopping if stopping is not None else f"{broken[0]} {status[broken[0]]}")
                    continue
                ready = all(key in status for key in job.after)
                if not ready or held_back or need[job.key] > len(free):
                    held_back = held_back or ready
                    still.append(job)
                    continue
                places = sorted(free)[:need[job.key]]
                started = self._start(job, places)
                if started is None:
                    status[job.key] = FAILED
                else:
                    free.difference_update(places)
                    running[job.key] = started
            waiting = still
            if not running:
                break                               # a waiting job would have started: nothing is left

            ending = [entry for entry in running.values() if entry.cause is not None]
            try:
                message = self._inbox.get(timeout=self.grace if ending else None)
            except queue.Empty:                     # children that ignore the request to end
                for entry in ending:
                    entry.stop(entry.cause, hard=True)
                continue
            if message[0] == "stop":
                hard = stopping is not None         # a second request does not wait
                if not hard:
                    stopping = message[1]
                    self.record("stop", reason=stopping)
                for entry in running.values():
                    entry.stop(STOPPED, hard)
                continue
            _, key, name, code = message
            entry = running[key]
            entry.codes[name] = code
            if code != 0:
                entry.stop(FAILED)                  # the tasks of a job stop together
            if len(entry.codes) == len(entry.handles):
                del running[key]
                free.update(entry.places)
                failed = any(entry.codes.values())
                status[key] = (entry.cause or FAILED) if failed else DONE
                self.record("end", job=key, devices=entry.devices, status=status[key], returncodes=entry.codes)
        self.record("finish", status=status)
        return status


class ScheduleLog:
    """Numbers the events of one call and appends them to a file (and to ``events``)."""

    def __init__(self, path=None, events: Optional[Callable[[dict], None]] = None):
        self.path, self.events, self.seq = path, events, 0

    def __call__(self, event: str, **fields) -> None:
        entry = dict(seq=self.seq, time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     event=event, **fields)
        self.seq += 1
        if self.path is not None:
            with open(self.path, "a") as handle:
                handle.write(json.dumps(entry) + "\n")
        if self.events is not None:
            self.events(entry)


def run_jobs(jobs: Sequence[Job], devices: Sequence, launcher=None, log=None,
             events: Optional[Callable[[dict], None]] = None) -> dict:
    """Run ``jobs`` on ``devices`` and return the status of each job.

    The record of the call is appended to the file ``log`` and passed to ``events``. In the main
    thread, SIGINT and SIGTERM stop the children before the call returns."""
    scheduler = Scheduler(jobs, devices, launcher, ScheduleLog(log, events))
    if threading.current_thread() is not threading.main_thread():
        return scheduler.run()                      # signal handlers belong to the main thread

    def stop(signum, frame):
        scheduler.request_stop(signal.Signals(signum).name)

    previous = {signum: signal.signal(signum, stop) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        return scheduler.run()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@contextmanager
def directory_lock(directory):
    """Hold ``directory`` for this process; a second holder gets a ``RuntimeError``.

    The lock is the kernel's (``flock``): it ends with the process, however that ends, and it
    is seen by every container that mounts the directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / ".lock", "a") as owner, open(directory / ".running", "a") as marker:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"{directory} is in use by another fmeta process") from None
        # Only is_locked() shares the marker with an owner, and only for an instant: waiting
        # here cannot last, and a look at the state can never make a run fail to start.
        fcntl.flock(marker, fcntl.LOCK_EX)
        yield


def is_locked(directory) -> bool:
    """Whether a process holds ``directory`` right now (see :func:`directory_lock`)."""
    try:
        with open(Path(directory) / ".running") as marker:
            try:
                fcntl.flock(marker, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
    except FileNotFoundError:
        pass
    return False


def read_schedule(path) -> list:
    """The records in a ``schedule.jsonl``, one list of events per call (oldest first)."""
    calls: list = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event["event"] == "begin" or not calls:
                calls.append([])
            calls[-1].append(event)
    return calls


def verify_schedule(events: Iterable[dict]) -> list:
    """What the record of one call shows against the rules of the scheduler (empty: nothing)."""
    events = list(events)
    if not events or events[0]["event"] != "begin":
        return ["the record does not start with a begin event"]
    problems = []
    if [event["seq"] for event in events] != list(range(len(events))):
        problems.append("the sequence numbers are not 0, 1, 2, ...")
    jobs = {job["key"]: job for job in events[0]["jobs"]}
    free = list(events[0]["devices"])
    running: dict = {}
    ended: dict = {}
    skipped: set = set()
    stopped = False
    for event in events[1:]:
        kind, key = event["event"], event.get("job")
        if kind in ("start", "skip") and (key not in jobs or key in running or key in ended or key in skipped):
            problems.append(f"{key} is not a job of this call or was settled before")
        elif kind == "start":
            if stopped:
                problems.append(f"{key} started after the stop request")
            for other in jobs[key]["after"]:
                if ended.get(other) != DONE:
                    problems.append(f"{key} started before {other} had ended successfully")
            taken = event["devices"]
            if len(taken) != jobs[key]["devices"]:
                problems.append(f"{key} asked for {jobs[key]['devices']} device(s) and was given {taken}")
            for device in taken:
                if device in free:
                    free.remove(device)
                else:
                    problems.append(f"{key} was given device {device}, which was not free")
            if sorted(event["tasks"]) != sorted(jobs[key]["tasks"]) or any(
                    device not in taken for device in event["tasks"].values()):
                problems.append(f"the tasks of {key} are not the job's tasks on the job's devices")
            running[key] = taken
        elif kind == "skip":
            skipped.add(key)
        elif kind == "end":
            if key not in running:
                problems.append(f"{key} ended without having started")
            else:
                free.extend(running.pop(key))
                ended[key] = event["status"]
        elif kind == "stop":
            stopped = True
    if events[-1]["event"] != "finish":
        problems.append("the record has no finish event: the scheduler was killed or is still running")
    else:
        unsettled = sorted(set(jobs) - set(ended) - skipped)
        if running or unsettled:
            problems.append(f"at the end {sorted(running)} were running and {unsettled} had neither run "
                            "nor been skipped")
    return problems
