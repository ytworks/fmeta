"""The scheduler of the replicas (FM-D14): the order of its decisions.

The children are objects that end when the test ends them, so the order of the events is the
scheduler's alone. ``HANG`` only ends a test that would wait forever."""

import os
import queue
import threading
from pathlib import Path

import pytest

from fmeta.config import Config
from fmeta.engine.platform import resolve_devices
from fmeta.scheduler import (
    Job,
    ScheduleLog,
    Scheduler,
    Task,
    directory_lock,
    is_locked,
    read_schedule,
    verify_schedule,
)
from fmeta.workflow import run_replicas

HANG = 60.0


class Child:
    """Ends when the test ends it, or when the scheduler terminates it (unless it ignores that)
    or kills it."""

    def __init__(self, ignores_terminate=False):
        self.pid, self.ignores_terminate, self.signals = None, ignores_terminate, []
        self._codes = queue.SimpleQueue()

    def end(self, code=0):
        self._codes.put(code)

    def wait(self):
        return self._codes.get(timeout=HANG)

    def terminate(self):
        self.signals.append("terminate")
        if not self.ignores_terminate:
            self.end(-15)

    def kill(self):
        self.signals.append("kill")
        self.end(-9)


def job(key, tasks=None, **kwargs):
    return Job(key, [Task(name, [], Path(os.devnull)) for name in tasks or [key]], **kwargs)


def brief(event):
    kind = event["event"]
    if kind in ("start", "end", "skip"):
        return kind, event["job"], event[{"start": "devices", "end": "status", "skip": "reason"}[kind]]
    return kind, event.get("reason", event.get("status"))


class Run:
    """A scheduler in a thread; ``take`` returns its next events in their short form."""

    def __init__(self, jobs, devices, grace=HANG, cannot_start=(), ignore_terminate=()):
        self.children, self.record, self._events = {}, [], queue.SimpleQueue()

        def launcher(task, device):
            if task.name in cannot_start:
                raise OSError(f"no program for {task.name}")
            child = self.children[task.name] = Child(task.name in ignore_terminate)
            return child

        self.scheduler = Scheduler(jobs, devices, launcher, ScheduleLog(events=self._events.put), grace)
        threading.Thread(target=self.scheduler.run, daemon=True).start()
        assert self.take(1) == [("begin", None)]

    def take(self, count):
        events = [self._events.get(timeout=HANG) for _ in range(count)]
        self.record += events
        return [brief(event) for event in events]


def test_three_jobs_on_two_devices_the_third_starts_on_the_released_device():
    run = Run([job("a"), job("b"), job("c")], devices=[4, 7])
    assert run.take(2) == [("start", "a", [4]), ("start", "b", [7])]
    run.children["b"].end()
    assert run.take(2) == [("end", "b", "done"), ("start", "c", [7])]
    run.children["c"].end()
    assert run.take(1) == [("end", "c", "done")]
    run.children["a"].end()
    assert run.take(2) == [("end", "a", "done"), ("finish", {"a": "done", "b": "done", "c": "done"})]
    assert verify_schedule(run.record) == []


def test_a_device_named_twice_carries_two_jobs_side_by_side():
    run = Run([job("a"), job("b"), job("c")], devices=[0, 0])
    assert run.take(2) == [("start", "a", [0]), ("start", "b", [0])]
    run.children["a"].end()
    assert run.take(2) == [("end", "a", "done"), ("start", "c", [0])]
    run.children["b"].end()
    assert run.take(1) == [("end", "b", "done")]
    run.children["c"].end()
    assert run.take(2) == [("end", "c", "done"), ("finish", {"a": "done", "b": "done", "c": "done"})]
    assert verify_schedule(run.record) == []


def test_a_failed_task_stops_its_job_and_what_comes_after_it_while_other_jobs_go_on():
    jobs = [job("rep0", ["w0", "w1"], devices=2), job("next", after=["rep0"]), job("other")]
    run = Run(jobs, devices=[0, 1, 2])
    assert run.take(2) == [("start", "rep0", [0, 1]), ("start", "other", [2])]
    assert run.record[1]["tasks"] == {"w0": 0, "w1": 1}
    run.children["w1"].end(1)
    assert run.take(2) == [("end", "rep0", "failed"), ("skip", "next", "rep0 failed")]
    assert run.record[3]["returncodes"] == {"w1": 1, "w0": -15}        # the other walker was terminated
    run.children["other"].end()
    assert run.take(2) == [("end", "other", "done"),
                           ("finish", {"rep0": "failed", "next": "skipped", "other": "done"})]
    assert verify_schedule(run.record) == []


def test_a_job_that_waits_for_devices_holds_back_the_jobs_behind_it():
    run = Run([job("a"), job("wide", ["w0", "w1"], devices=2), job("c")], devices=[0, 1])
    assert run.take(1) == [("start", "a", [0])]
    run.children["a"].end()
    assert run.take(2) == [("end", "a", "done"), ("start", "wide", [0, 1])]    # c did not take device 1
    run.children["w0"].end()
    run.children["w1"].end()
    assert run.take(2) == [("end", "wide", "done"), ("start", "c", [0])]
    run.children["c"].end()
    assert run.take(2)[0] == ("end", "c", "done")
    assert verify_schedule(run.record) == []


@pytest.mark.parametrize("second_request", [True, False])
def test_after_a_stop_request_nothing_starts_and_a_child_that_ignores_it_is_killed(second_request):
    """The child is killed at the second request, or when the grace time is over."""
    run = Run([job("a"), job("b")], devices=[0], ignore_terminate=["a"],
              grace=HANG if second_request else 0.01)
    assert run.take(1) == [("start", "a", [0])]
    run.scheduler.request_stop("SIGTERM")
    assert run.take(2) == [("stop", "SIGTERM"), ("skip", "b", "SIGTERM")]
    if second_request:
        run.scheduler.request_stop("SIGTERM")
    assert run.take(2) == [("end", "a", "stopped"), ("finish", {"a": "stopped", "b": "skipped"})]
    assert run.children["a"].signals[:2] == ["terminate", "kill"]
    assert verify_schedule(run.record) == []


def test_a_job_that_cannot_start_fails_and_leaves_no_child_behind():
    jobs = [job("rep0", ["w0", "w1"], devices=2), job("next", after=["rep0"]), job("other")]
    run = Run(jobs, devices=[0, 1], cannot_start=["w1"])
    assert run.take(3) == [("skip", "rep0", "could not start: no program for w1"),
                           ("skip", "next", "rep0 failed"), ("start", "other", [0])]
    assert run.children["w0"].signals == ["kill"]
    run.children["other"].end()
    assert run.take(2) == [("end", "other", "done"),
                           ("finish", {"rep0": "failed", "next": "skipped", "other": "done"})]
    assert verify_schedule(run.record) == []


def _record(after, *events):
    begin = dict(event="begin", devices=[0, 1],
                 jobs=[dict(key="a", tasks=["a"], devices=1, after=[]),
                       dict(key="b", tasks=["b"], devices=1, after=after)])
    return [dict(event, seq=seq) for seq, event in enumerate((begin,) + events)]


def _start(key, device):
    return dict(event="start", job=key, devices=[device], tasks={key: device})


def _end(key, status="done"):
    return dict(event="end", job=key, status=status)


_STOP, _FINISH = dict(event="stop", reason="SIGTERM"), dict(event="finish")


@pytest.mark.parametrize("record, problem", [
    (_record([], _start("a", 0), _start("b", 0), _end("a"), _end("b"), _FINISH),
     "b was given device 0, which was not free"),
    (_record(["a"], _start("a", 0), _end("a", "failed"), _start("b", 0), _end("b"), _FINISH),
     "b started before a had ended successfully"),
    (_record([], _start("a", 0), _STOP, _start("b", 1), _end("a", "stopped"), _end("b"), _FINISH),
     "b started after the stop request"),
    (_record([], _start("a", 0), _end("a"), _FINISH),
     "at the end [] were running and ['b'] had neither run nor been skipped"),
    (_record([], _start("a", 0), _start("b", 1), _end("a")),
     "the record has no finish event: the scheduler was killed or is still running"),
])
def test_a_record_that_breaks_a_rule_of_the_scheduler_is_reported(record, problem):
    assert verify_schedule(record) == [problem]


@pytest.mark.parametrize("mode, gpus, visible, devices", [
    ("auto", None, 4, [None]),                  # one device, the platform's own choice
    ("auto", [2, 3], 4, [2, 3]),
    ("sequential", [3], 4, [3]),
    ("parallel", None, 4, [0, 1, 2, 3]),
    ("parallel", "all", 2, [0, 1]),
    ("parallel", [1, 0], 4, [1, 0]),
])
def test_mode_and_gpus_give_the_devices_of_the_replicas(mode, gpus, visible, devices):
    assert resolve_devices(mode, gpus, "CUDA", visible=lambda: [{}] * visible) == devices


def test_per_gpu_names_every_device_that_many_times_and_the_first_replicas_get_their_own():
    assert resolve_devices("parallel", [0, 1], "CUDA", visible=lambda: [{}] * 2, per_gpu=2) == [0, 1, 0, 1]
    assert resolve_devices("auto", None, "CUDA", per_gpu=3) == [None, None, None]
    with pytest.raises(ValueError, match="leave run.per_gpu at 1"):
        resolve_devices("sequential", [0], "CUDA", per_gpu=2)


def test_devices_that_are_not_there_are_refused():
    with pytest.raises(ValueError, match="device 4 is not there: nvidia-smi lists 4 GPU"):
        resolve_devices("parallel", [0, 4], "CUDA", visible=lambda: [{}] * 4)
    with pytest.raises(ValueError, match="needs GPUs that nvidia-smi lists"):
        resolve_devices("parallel", "all", "CUDA", visible=lambda: None)


def test_a_directory_is_held_by_one_run_at_a_time(tmp_path):
    assert not is_locked(tmp_path)
    with directory_lock(tmp_path):
        with pytest.raises(RuntimeError, match="is in use by another fmeta process"):
            with directory_lock(tmp_path):
                pass
        assert is_locked(tmp_path)              # the refused attempt did not release it
    assert not is_locked(tmp_path)


def test_run_replicas_names_the_replica_that_failed_and_the_last_line_of_its_log(tmp_path):
    config = Config.from_dict({"run": {"replicas": 3, "platform": "Reference"}})

    def launcher(task, device):
        child = Child()
        if task.name == "rep1":
            Path(task.log).write_text("Traceback (most recent call last):\nfmeta: error: the bias is not finite\n")
        child.end(1 if task.name == "rep1" else 0)
        return child

    with pytest.raises(RuntimeError, match=r"^rep1 failed: the bias is not finite \(see .*rep1\.log\)$"):
        run_replicas(config, tmp_path, launcher=launcher)
    record = read_schedule(tmp_path / "schedule.jsonl")[-1]
    assert verify_schedule(record) == []
    assert record[-1]["status"] == {"rep0": "done", "rep1": "failed", "rep2": "done"}


def test_run_replicas_gives_each_walker_of_a_replica_its_own_device(tmp_path):
    config = Config.from_dict({"run": {"replicas": 2, "walkers": 2, "platform": "Reference", "gpus": [0, 1]}})
    started = []

    def launcher(task, device):
        started.append((task.name, device))
        child = Child()
        child.end()
        return child

    run_replicas(config, tmp_path, launcher=launcher)
    assert started == [("rep0_w0", 0), ("rep0_w1", 1), ("rep1_w0", 0), ("rep1_w1", 1)]
