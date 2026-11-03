"""tools/pulsatile_supervisor.py: crash detection, pid re-checks, one relaunch, report joining.

Fake psutil processes and a fake clock only: no real process is started, signalled or ended here.
"""
import csv
import datetime as dt
import json
import os
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))
import pulsatile_supervisor as S      # noqa: E402

ME = "HP-WORKSTATION\\Vishesh"
T0 = dt.datetime(2026, 10, 7, 12, 30, 0)
UNTIL = dt.datetime(2026, 10, 7, 21, 45)
HDR = "step,flow_time_s,tau_s,cycle,iterations,capped,step_wall_s\n"


# ---------------------------------------------------------------- fakes
class FakeClock:
    def __init__(self, t):
        self.t = t
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += dt.timedelta(seconds=s)

    def advance(self, **kw):
        self.t += dt.timedelta(**kw)


class NoSuchProcess(Exception):
    pass


class AccessDenied(Exception):
    pass


class FakeProc:
    def __init__(self, ps, pid, name, ct, user=ME, deny=False, stubborn=False, cmd=None):
        self.ps, self.pid, self._name, self._ct, self._user = ps, pid, name, ct, user
        self.deny, self.stubborn, self.alive, self.kids = deny, stubborn, True, []
        self._cmd = cmd or [name]

    def cmdline(self):
        self._chk()
        return list(self._cmd)

    def _chk(self):
        if not self.alive:
            raise NoSuchProcess(self.pid)
        if self.deny:
            raise AccessDenied(self.pid)

    def name(self):
        self._chk()
        return self._name

    def create_time(self):
        self._chk()
        return self._ct

    def username(self):
        self._chk()
        if self._user is None:
            raise AccessDenied(self.pid)
        return self._user

    def children(self, recursive=False):
        self._chk()
        out = []
        for k in self.kids:
            if k.alive:
                out.append(k)
                if recursive:
                    out += k.children(recursive=True)
        return out

    def is_running(self):
        return self.alive

    def kill(self):
        if not self.alive:
            raise NoSuchProcess(self.pid)
        self.ps.killed.append(self.pid)
        if not self.stubborn:
            self.alive = False


class FakePs:
    NoSuchProcess = NoSuchProcess
    AccessDenied = AccessDenied

    def __init__(self):
        self.procs, self.killed = {}, []

    def add(self, pid, name, ct, **kw):
        p = FakeProc(self, pid, name, ct, **kw)
        self.procs[pid] = p
        return p

    def Process(self, pid=None):                                # noqa: N802 (psutil API)
        if pid is None:
            return types.SimpleNamespace(pid=1, username=lambda: ME)
        p = self.procs.get(pid)
        if p is None or not p.alive:
            raise NoSuchProcess(pid)
        return p

    def wait_procs(self, procs, timeout=None):
        for p in procs:
            if getattr(p, "dies_on_wait", False):                # e.g. a hold process: exits after its parent
                p.alive = False
        return [p for p in procs if not p.alive], [p for p in procs if p.alive]

    def process_iter(self):
        return [p for p in list(self.procs.values()) if p.alive]


class FakePopen:
    def __init__(self, ps, clock, pid=7001):
        self.ps, self.clock, self.pid, self.calls = ps, clock, pid, []

    def __call__(self, argv, **kw):
        self.calls.append((list(argv), kw))
        out = kw.get("stdout")
        name = getattr(out, "name", None)
        if hasattr(out, "close"):
            out.close()
        pid = self.pid + len(self.calls) - 1
        self.ps.add(pid, "python.exe", self.clock.now().timestamp())
        return types.SimpleNamespace(pid=pid, stdout_name=name)


# ---------------------------------------------------------------- run-folder helpers
def ts(t):
    return t.timestamp()


def write_pids(rd, attempt, procs):
    (rd / "pids.json").write_text(json.dumps({"attempt": attempt, "processes": procs}), encoding="utf-8")


def rec(role, p):
    return {"role": role, "pid": p.pid, "image": p._name, "create_time": p._ct}


def set_status(rd, state):
    (rd / "status.json").write_text(json.dumps({"state": state}), encoding="utf-8")


def add_rows(path, first, n, when, dt_s=0.005):
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        if new:
            fh.write(HDR)
        for k in range(first, first + n):
            fh.write(f"{k},{round(k * dt_s, 9)},0,1,10,False,60\n")
    os.utime(path, (ts(when), ts(when)))


def touch(path, when):
    os.utime(path, (ts(when), ts(when)))


def make_run(tmp_path, clk, n_nodes=2, attempt=1, launch=None, with_nodes=True):
    """A live attempt: run script, launcher, front-end and n solver nodes, all alive."""
    ps = FakePs()
    launch = launch or clk.now()
    c = ts(launch)
    base = 1000 * attempt
    pr = {"run_script": ps.add(base + 1, "python.exe", c), "launcher": ps.add(base + 2, "fluent.exe", c + 3),
          "frontend": ps.add(base + 3, "cx2610.exe", c + 4)}
    nodes = [ps.add(base + 10 + i, "fl_mpi2610.exe", c + 20 + i) for i in range(n_nodes)]
    procs = [rec(r, p) for r, p in pr.items()] + ([rec("node", n) for n in nodes] if with_nodes else [])
    write_pids(tmp_path, attempt, procs)
    set_status(tmp_path, "SOLVING")
    return ps, pr, nodes


def make_sup(tmp_path, ps, clk, popen=None, disk=150.0, relaunch_fn=None, before=None, **kw):
    calls = []

    def stub(run_dir, design, nproc, cpus):
        calls.append((run_dir, design, nproc, cpus))
        save = tmp_path / "saves" / "sine_cf2_020_rs_000240.dat.h5"
        return ["C:/env/python.exe", "-u", "tools/pulsatile_top3.py", "run", "--design", design,
                "--nproc", str(nproc), "--cpus", cpus, "--resume-data", str(save)]

    sup = S.Supervisor(design="sine_cf2_020", run_dir=tmp_path, nproc=16, cpus="20-35", until=UNTIL,
                       ps=ps, clock=clk.now, sleep=clk.sleep, disk_free_gib=lambda: disk,
                       popen=popen or FakePopen(ps, clk), relaunch_fn=relaunch_fn or stub, me_user=ME,
                       relaunch_before=before or dt.datetime(2026, 10, 7, 21, 0), **kw)
    sup.stub_calls = calls
    return sup


def write_index(rd):
    (rd / "saves").mkdir(exist_ok=True)
    idx = [{"step": 220, "flow_time": 1.1, "last_reset_t": 0.8, "sampled_time": 0.3,
            "path": str(rd / "saves" / "sine_cf2_020_rs_000220.dat.h5"), "size": 1, "sha256": "x", "kind": "rolling"},
           {"step": 240, "flow_time": 1.2, "last_reset_t": 0.8, "sampled_time": 0.4,
            "path": str(rd / "saves" / "sine_cf2_020_rs_000240.dat.h5"), "size": 1, "sha256": "y", "kind": "rolling"},
           {"step": 160, "flow_time": 0.8, "last_reset_t": 0.8, "sampled_time": 0.0,
            "path": str(rd / "saves" / "sine_cf2_020_cyc_000160.dat.h5"), "size": 1, "sha256": "not hashed",
            "kind": "cycle"}]
    (rd / "saves" / "index.json").write_text(json.dumps(idx), encoding="utf-8")


def state(rd):
    return json.loads((rd / "supervisor_state.json").read_text(encoding="utf-8"))


def log(rd):
    return (rd / "supervisor.log").read_text(encoding="utf-8")


def healthy_step(rd, clk, first, n=3):
    add_rows(rd / "pt_steps.csv", first, n, clk.now())


def first_poll(sup, rd, clk, minutes=40):
    """The supervisor starts with its run: a healthy first poll takes the evidence baseline (text already on disk at
    the first poll is never crash evidence)."""
    clk.advance(minutes=minutes)
    healthy_step(rd, clk, 1)
    assert sup.poll() == "continue"


# ---------------------------------------------------------------- constants
def test_constants_match_the_ruling():
    assert S.POLL_S == 300 and S.NO_ROW_S == 20 * 60 and S.GRACE_S == 30 * 60
    assert S.C_FREE_MIN_GIB == 123.0 and S.MAX_END_POLLS == 3
    assert S.STEPS_NAME == "pt_steps.csv" and S.A2_STEPS_NAME == "pt_steps_a2.csv"
    assert set(S.FINAL_STATES) == {"DONE_PERIODIC", "DONE_NOT_PERIODIC", "DONE_4_CYCLES", "STOPPED_end_time",
                                   "STOPPED_disk", "STOPPED_massflow_check", "NOT RUN", "FAILED"}
    assert S.DETACH_FLAGS == 0x8 | 0x200 | 0x4000


# ---------------------------------------------------------------- healthy run
def test_healthy_poll_logs_every_criterion_and_takes_no_action(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    sup = make_sup(tmp_path, ps, clk)
    clk.advance(minutes=10)
    healthy_step(tmp_path, clk, 1)
    assert sup.poll() == "continue"
    clk.advance(minutes=5)
    healthy_step(tmp_path, clk, 4)
    assert sup.poll() == "continue"
    text = log(tmp_path)
    lines = [ln for ln in text.splitlines() if " verdict " in ln]
    assert len(lines) == 2
    for ln in lines:
        dt.datetime.fromisoformat(ln.split()[0])                  # every poll line starts with a timestamp
        assert "verdict OK" in ln and "(a)" in ln and "(b)" in ln and "(c)" in ln
    assert "grace" in text and "2/2 recorded node pids alive" in text
    assert ps.killed == [] and sup.popen.calls == []
    assert state(tmp_path)["attempt"] == 1 and state(tmp_path)["ended"] is None


@pytest.mark.parametrize("final", sorted(["DONE_PERIODIC", "DONE_NOT_PERIODIC", "DONE_4_CYCLES", "STOPPED_end_time",
                                          "STOPPED_disk", "STOPPED_massflow_check", "NOT RUN", "FAILED"]))
def test_final_state_is_not_a_crash_and_the_supervisor_exits(tmp_path, final):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    for p in list(pr.values()) + nodes:
        p.alive = False                                          # Fluent exited at the end of the run
    set_status(tmp_path, final)
    sup = make_sup(tmp_path, ps, clk)
    clk.advance(minutes=50)
    assert sup.poll() == "exit"
    assert ps.killed == [] and sup.popen.calls == []
    assert f"final state {final}" in log(tmp_path)


def test_end_of_run_race_nodes_gone_then_final_state_within_the_confirmation_is_no_crash(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    clk.advance(minutes=40)
    healthy_step(tmp_path, clk, 1)
    for n in nodes:
        n.alive = False                                          # solver.exit() done, status not yet written
    real_sleep = clk.sleep

    def sleep_then_final(s):
        real_sleep(s)
        set_status(tmp_path, "DONE_PERIODIC")

    sup = make_sup(tmp_path, ps, clk)
    sup.sleep = sleep_then_final
    assert sup.poll() == "exit"
    assert ps.killed == [] and sup.popen.calls == []
    assert "final state DONE_PERIODIC" in log(tmp_path)


# ---------------------------------------------------------------- (a) crash evidence
def test_abnormal_exit_crash_ends_pids_in_order_then_relaunches_once(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk, n_nodes=3)
    write_index(tmp_path)
    sup = make_sup(tmp_path, ps, clk)
    clk.advance(minutes=40)
    healthy_step(tmp_path, clk, 1)
    assert sup.poll() == "continue"
    clk.advance(minutes=5)
    (tmp_path / "fluent-0-error.log").write_text("\nError [node 0] [time 10/7/26 13:15:02] Abnormal Exit!\n")
    nodes[0].alive = False                                       # one node died; two hang on (still alive)
    assert sup.poll() == "continue"                              # crashed, ended, relaunched -> watching attempt 2
    want = [pr["run_script"].pid, pr["launcher"].pid, pr["frontend"].pid, nodes[1].pid, nodes[2].pid]
    assert ps.killed == want                                     # run script, launcher, frontend, then nodes only if alive
    assert len(sup.popen.calls) == 1
    argv, kw = sup.popen.calls[0]
    assert argv[:4] == ["C:/env/python.exe", "-u", "tools/pulsatile_top3.py", "run"]
    assert kw["creationflags"] == S.DETACH_FLAGS
    assert kw["cwd"] == str(S.REPO)
    assert Path(kw["stdout"].name).name == "attempt2_wrapper.out"
    assert kw["env"]["STENOSIS_FLUENT_NPROC"] == "16"
    assert sup.stub_calls == [(tmp_path, "sine_cf2_020", 16, "20-35")]
    st = state(tmp_path)
    assert st["attempt"] == 2 and st["ended"] is None
    r = st["relaunch"]
    assert r["command"] == argv and r["save"]["step"] == 240 and r["resumed_from_step"] == 240
    assert r["save"]["flow_time"] == 1.2 and r["time"].startswith("2026-10-07T13:")
    assert r["pid"] == 7001 and r["image"] == "python.exe"
    text = log(tmp_path)
    assert "CRASHED" in text and "Abnormal Exit!" in text and "fluent-0-error.log" in text
    assert "RELAUNCHED" in text


def test_critical_code_section_in_the_transcript_is_a_crash(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    (tmp_path / "pt_transcript.txt").write_text("iter continuity\n  1 1e-3\n")
    sup = make_sup(tmp_path, ps, clk)
    clk.advance(minutes=40)
    healthy_step(tmp_path, clk, 1)
    assert sup.poll() == "continue"
    with (tmp_path / "pt_transcript.txt").open("a") as fh:
        fh.write("Error: Unknown object encountered in garbage collection\nError encountered in critical code section\n")
    clk.advance(minutes=5)
    sup.poll()
    assert "CRASHED" in log(tmp_path) and "critical code section" in log(tmp_path)
    assert pr["run_script"].pid in ps.killed and len(sup.popen.calls) == 1


def test_overnight_evidence_already_in_a_reused_folder_is_ignored(tmp_path):
    clk = FakeClock(T0)
    old = "\nError [node 0] [time 10/7/26 2:48:52] Abnormal Exit!\n"
    (tmp_path / "fluent-0-error.log").write_text(old)
    (tmp_path / "fluent-999999-error.log").write_text(old)
    (tmp_path / "pt_transcript.txt").write_text("Error encountered in critical code section\n")
    (tmp_path / "fluent-20261006-205439-57140.trn").write_text("Error encountered in critical code section\n")
    ps, pr, nodes = make_run(tmp_path, clk)
    sup = make_sup(tmp_path, ps, clk)
    clk.advance(minutes=10)
    healthy_step(tmp_path, clk, 1)
    with (tmp_path / "pt_transcript.txt").open("a") as fh:     # PyFluent appends to the same transcript
        fh.write("  12 1.0e-05 healthy\n")
    assert sup.poll() == "continue"
    clk.advance(minutes=5)
    healthy_step(tmp_path, clk, 4)
    assert sup.poll() == "continue"
    assert ps.killed == [] and "CRASHED" not in log(tmp_path)
    (tmp_path / "pt_transcript.txt").write_text("rewritten\nError encountered in critical code section\n")
    clk.advance(minutes=5)
    healthy_step(tmp_path, clk, 7)
    sup.poll()                                                   # a rewritten (shorter) file is scanned from 0
    assert "CRASHED" in log(tmp_path)


# ---------------------------------------------------------------- (b) solver nodes
def test_all_recorded_nodes_gone_is_a_crash_partial_loss_only_warns(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk, n_nodes=2)
    sup = make_sup(tmp_path, ps, clk)
    clk.advance(minutes=10)
    healthy_step(tmp_path, clk, 1)
    nodes[0].alive = False
    assert sup.poll() == "continue"
    assert "1/2 recorded node pids alive" in log(tmp_path) and ps.killed == []
    nodes[1].alive = False
    clk.advance(minutes=5)
    healthy_step(tmp_path, clk, 4)
    sup.poll()
    assert "CRASHED" in log(tmp_path) and "node" in log(tmp_path)
    assert ps.killed == [pr["run_script"].pid, pr["launcher"].pid, pr["frontend"].pid]


def test_reused_node_pid_counts_as_gone_and_is_never_ended(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk, n_nodes=1)
    sup = make_sup(tmp_path, ps, clk)
    clk.advance(minutes=10)
    healthy_step(tmp_path, clk, 1)
    sup.poll()
    nodes[0].alive = False
    imposter = ps.add(nodes[0].pid, "fl_mpi2610.exe", ts(clk.now()))   # same pid + image, later start: reused
    clk.advance(minutes=5)
    healthy_step(tmp_path, clk, 4)
    sup.poll()
    assert "CRASHED" in log(tmp_path)
    assert imposter.pid not in ps.killed and imposter.alive
    assert "create_time" in log(tmp_path) and "SKIP" in log(tmp_path)


def test_before_nodes_are_recorded_the_launcher_tree_is_used(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk, n_nodes=0, with_nodes=False)
    host = ps.add(4242, "fl2610.exe", ts(T0) + 30)
    pr["launcher"].kids = [pr["frontend"]]
    pr["frontend"].kids = [host]
    sup = make_sup(tmp_path, ps, clk)
    clk.advance(minutes=5)
    assert sup.poll() == "continue"
    assert "launcher tree" in log(tmp_path) and "fl2610.exe" in log(tmp_path)
    host.alive = False
    clk.advance(minutes=5)
    sup.poll()
    assert "CRASHED" in log(tmp_path)
    assert ps.killed[:3] == [pr["run_script"].pid, pr["launcher"].pid, pr["frontend"].pid]


def test_launcher_gone_before_nodes_are_recorded_is_a_crash(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk, n_nodes=0, with_nodes=False)
    sup = make_sup(tmp_path, ps, clk)
    clk.advance(minutes=5)
    assert sup.poll() == "continue"
    assert "no solver process in its tree yet" in log(tmp_path)
    pr["launcher"].alive = False
    clk.advance(minutes=5)
    sup.poll()
    assert "CRASHED" in log(tmp_path) and "launcher" in log(tmp_path)


def test_no_pids_json_yet_means_no_verdict_and_no_action(tmp_path):
    clk = FakeClock(T0)
    ps = FakePs()
    set_status(tmp_path, "SOLVING")
    sup = make_sup(tmp_path, ps, clk)
    for _ in range(12):                                          # 60 min, nothing ever recorded
        clk.advance(minutes=5)
        assert sup.poll() == "continue"
    assert "pids.json" in log(tmp_path) and "CRASHED" not in log(tmp_path)
    assert sup.popen.calls == []


# ---------------------------------------------------------------- (c) no new row
def test_no_new_row_for_20_min_is_a_crash_only_after_the_30_min_grace(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    sup = make_sup(tmp_path, ps, clk)
    clk.advance(minutes=4)
    healthy_step(tmp_path, clk, 1, n=1)                          # last row at 12:34
    clk.advance(minutes=21)                                      # 12:55: 21 min without a row, but in the grace
    assert sup.poll() == "continue"
    assert "in grace" in log(tmp_path) and "CRASHED" not in log(tmp_path)
    clk.advance(minutes=6)                                       # 13:01: grace over, 27 min without a row
    sup.poll()
    assert "CRASHED" in log(tmp_path) and "(c)" in log(tmp_path)


def test_rows_keep_coming_no_crash_even_with_long_steps(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    sup = make_sup(tmp_path, ps, clk)
    k = 1
    for _ in range(20):                                          # 100 min, a row every 15 min (slow capped steps)
        clk.advance(minutes=5)
        if clk.now().minute % 15 == 0:
            healthy_step(tmp_path, clk, k, n=1)
            k += 1
        assert sup.poll() == "continue"
    assert "CRASHED" not in log(tmp_path)


# ---------------------------------------------------------------- end + confirm + relaunch rules
def test_remaining_pid_blocks_relaunch_retries_and_gives_up_after_3_polls(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    pr["frontend"].stubborn = True                               # TerminateProcess "succeeds", process stays
    sup = make_sup(tmp_path, ps, clk)
    clk.advance(minutes=40)
    healthy_step(tmp_path, clk, 1)
    for n in nodes:
        n.alive = False
    assert sup.poll() == "continue"
    want = f"not relaunched: pid {pr['frontend'].pid} (cx2610.exe) still alive after termination"
    assert sup.popen.calls == [] and want in log(tmp_path)
    clk.advance(minutes=5)
    assert sup.poll() == "continue"
    clk.advance(minutes=5)
    assert sup.poll() == "exit"
    assert ps.killed.count(pr["frontend"].pid) == 3
    assert sum(ln.endswith(want) for ln in log(tmp_path).splitlines()) == 3      # once per end poll
    assert sup.popen.calls == []
    st = state(tmp_path)
    assert st["ended"] and "still alive" in st["ended"]


# ---------------------------------------------------------------- PyFluent watchdog first (lead, 11:31)
def wd_cmd(run_pid, secret="pw-SECRET-123"):
    return [r"C:\env\pythonw.exe", r"C:\env\Lib\site-packages\ansys\fluent\core\launcher\watchdog_exec",
            str(run_pid), "127.0.0.1", "50051", secret, "abc123", "False", "None", "False", "False"]


def test_pyfluent_watchdog_found_by_command_line_is_ended_before_the_run_script(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    rs = pr["run_script"]
    wd = ps.add(3001, "pythonw.exe", rs._ct + 40, cmd=wd_cmd(rs.pid))
    other_run = ps.add(3002, "pythonw.exe", rs._ct + 40, cmd=wd_cmd(rs.pid + 777))
    overnight = ps.add(3003, "pythonw.exe", rs._ct - 3600 * 10, cmd=wd_cmd(rs.pid))   # older than the run script
    sup = make_sup(tmp_path, ps, clk)
    clk.advance(minutes=40)
    healthy_step(tmp_path, clk, 1)
    assert sup.poll() == "continue"
    assert "pyfluent watchdog pid 3001" in log(tmp_path)
    (tmp_path / "fluent-0-error.log").write_text("Abnormal Exit!\n")
    clk.advance(minutes=5)
    sup.poll()
    assert ps.killed[:4] == [wd.pid, rs.pid, pr["launcher"].pid, pr["frontend"].pid]
    assert other_run.alive and overnight.alive and other_run.pid not in ps.killed and overnight.pid not in ps.killed
    assert "pw-SECRET-123" not in log(tmp_path) and "pw-SECRET-123" not in (tmp_path / "supervisor_state.json").read_text()
    assert len(sup.popen.calls) == 1


def test_pyfluent_watchdog_running_as_python_exe_is_found_too(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    rs = pr["run_script"]
    wd = ps.add(3401, "python.exe", rs._ct + 40, cmd=wd_cmd(rs.pid))   # no pythonw.exe in the env: PyFluent uses python
    sup = make_sup(tmp_path, ps, clk)
    first_poll(sup, tmp_path, clk)
    (tmp_path / "fluent-0-error.log").write_text("Abnormal Exit!\n")
    clk.advance(minutes=5)
    sup.poll()
    assert ps.killed[:2] == [wd.pid, rs.pid]


def test_pyfluent_watchdog_recorded_in_pids_json_is_ended_first(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    rs = pr["run_script"]
    wd = ps.add(3101, "pythonw.exe", rs._ct + 40, cmd=wd_cmd(rs.pid))
    doc = json.loads((tmp_path / "pids.json").read_text())
    doc["processes"].append(rec("pyfluent_watchdog", wd))
    (tmp_path / "pids.json").write_text(json.dumps(doc))
    sup = make_sup(tmp_path, ps, clk)
    first_poll(sup, tmp_path, clk)
    (tmp_path / "fluent-0-error.log").write_text("Abnormal Exit!\n")
    clk.advance(minutes=5)
    sup.poll()
    assert ps.killed[:2] == [wd.pid, rs.pid]


def test_more_than_one_watchdog_match_refuses_and_ends_nothing(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    rs = pr["run_script"]
    a = ps.add(3201, "pythonw.exe", rs._ct + 40, cmd=wd_cmd(rs.pid))
    b = ps.add(3202, "pythonw.exe", rs._ct + 41, cmd=wd_cmd(rs.pid))
    sup = make_sup(tmp_path, ps, clk)
    first_poll(sup, tmp_path, clk)
    (tmp_path / "fluent-0-error.log").write_text("Abnormal Exit!\n")
    clk.advance(minutes=5)
    sup.poll()
    assert ps.killed == [] and a.alive and b.alive and rs.alive
    assert "refusing to end any pid of attempt 1: 2 pyfluent watchdog matches" in log(tmp_path)
    assert sup.popen.calls == []
    for _ in range(2):
        clk.advance(minutes=5)
        sup.poll()
    assert ps.killed == [] and sup.popen.calls == [] and state(tmp_path)["ended"]


def test_watchdog_surviving_termination_keeps_the_run_script_alive(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    rs = pr["run_script"]
    wd = ps.add(3301, "pythonw.exe", rs._ct + 40, cmd=wd_cmd(rs.pid), stubborn=True)
    sup = make_sup(tmp_path, ps, clk)
    first_poll(sup, tmp_path, clk)
    (tmp_path / "fluent-0-error.log").write_text("Abnormal Exit!\n")
    clk.advance(minutes=5)
    sup.poll()
    assert ps.killed == [wd.pid] and rs.alive                    # never end the run script under a live watchdog
    assert f"not relaunched: pid {wd.pid} (pythonw.exe) still alive after termination" in log(tmp_path)
    assert sup.popen.calls == []


@pytest.mark.parametrize("disk,when,why", [(122.9, dt.datetime(2026, 10, 7, 14, 0), "C: free"),
                                           (200.0, dt.datetime(2026, 10, 7, 21, 0), "21:00")])
def test_relaunch_needs_123_gib_and_before_2100(tmp_path, disk, when, why):
    clk = FakeClock(when - dt.timedelta(minutes=45))
    ps, pr, nodes = make_run(tmp_path, clk)
    sup = make_sup(tmp_path, ps, clk, disk=disk)
    first_poll(sup, tmp_path, clk, minutes=38)
    (tmp_path / "fluent-0-error.log").write_text("Abnormal Exit!\n")
    clk.t = when - dt.timedelta(seconds=S.CONFIRM_S)             # CRASHED after the 2-min re-check, at `when`
    healthy_step(tmp_path, clk, 4)
    assert sup.poll() == "exit"
    assert pr["run_script"].pid in ps.killed                     # the crashed run is still ended
    assert sup.popen.calls == []
    assert "NOT relaunched" in log(tmp_path) and why in log(tmp_path)
    assert "not relaunched" in state(tmp_path)["ended"]


def test_second_crash_ends_the_design_without_another_relaunch(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    write_index(tmp_path)
    sup = make_sup(tmp_path, ps, clk)
    first_poll(sup, tmp_path, clk)
    (tmp_path / "fluent-0-error.log").write_text("Abnormal Exit!\n")
    clk.advance(minutes=5)
    assert sup.poll() == "continue"
    assert len(sup.popen.calls) == 1
    # attempt 2 runs: its own pids.json, rows in the new report file
    clk.advance(minutes=3)
    a2 = ps.procs[7001]
    l2 = ps.add(2002, "fluent.exe", ts(clk.now()))
    f2 = ps.add(2003, "cx2610.exe", ts(clk.now()))
    n2 = [ps.add(2010 + i, "fl_mpi2610.exe", ts(clk.now()) + 5) for i in range(2)]
    write_pids(tmp_path, 2, [rec("run_script", a2), rec("launcher", l2), rec("frontend", f2)]
               + [rec("node", n) for n in n2])
    set_status(tmp_path, "SOLVING")
    clk.advance(minutes=10)
    add_rows(tmp_path / S.A2_STEPS_NAME, 241, 3, clk.now())
    assert sup.poll() == "continue"
    assert "verdict OK" in log(tmp_path).splitlines()[-1]
    with (tmp_path / "fluent-0-error.log").open("a") as fh:
        fh.write("Abnormal Exit!\n")
    for n in n2:
        n.alive = False
    clk.advance(minutes=5)
    assert sup.poll() == "exit"
    assert len(sup.popen.calls) == 1                             # never a second relaunch
    assert ps.killed[-3:] == [a2.pid, l2.pid, f2.pid]
    assert state(tmp_path)["ended"] == "second crash"


def test_attempt2_never_judges_the_stale_attempt1_pids(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    sup = make_sup(tmp_path, ps, clk)
    first_poll(sup, tmp_path, clk)
    (tmp_path / "fluent-0-error.log").write_text("Abnormal Exit!\n")
    clk.advance(minutes=5)
    sup.poll()
    assert state(tmp_path)["attempt"] == 2
    n_log = len(log(tmp_path).splitlines())
    for _ in range(4):                                           # pids.json still says attempt 1 (all dead)
        clk.advance(minutes=5)
        assert sup.poll() == "continue"
    new = "\n".join(log(tmp_path).splitlines()[n_log:])
    assert "CRASHED" not in new and "attempt 2" in new and len(sup.popen.calls) == 1


def test_never_touches_shared_mpi_services_or_another_accounts_process(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    hydra = ps.add(5555, "hydra_service.exe", ts(T0))
    other = ps.add(5556, "fl_mpi2610.exe", ts(T0), user="HP-WORKSTATION\\someoneelse")
    doc = json.loads((tmp_path / "pids.json").read_text())
    doc["processes"] += [rec("node", hydra), rec("node", other)]
    (tmp_path / "pids.json").write_text(json.dumps(doc))
    sup = make_sup(tmp_path, ps, clk)
    first_poll(sup, tmp_path, clk)
    (tmp_path / "fluent-0-error.log").write_text("Abnormal Exit!\n")
    clk.advance(minutes=5)
    sup.poll()
    assert hydra.pid not in ps.killed and other.pid not in ps.killed
    assert hydra.alive and other.alive
    text = log(tmp_path)
    assert "shared MPI service" in text and "another account" in text
    assert sup.popen.calls == []                                 # their pids are alive: no relaunch


def test_until_ends_the_supervisor(tmp_path):
    clk = FakeClock(dt.datetime(2026, 10, 7, 21, 40))
    ps, pr, nodes = make_run(tmp_path, clk, launch=dt.datetime(2026, 10, 7, 12, 30))
    sup = make_sup(tmp_path, ps, clk)
    healthy_step(tmp_path, clk, 1)
    assert sup.poll() == "continue"
    clk.advance(minutes=5)
    assert sup.poll() == "exit"
    assert "until" in log(tmp_path)


def test_run_loop_polls_every_5_min_until_a_final_state(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    sup = make_sup(tmp_path, ps, clk)
    n = {"polls": 0}
    real = sup.poll

    def counted():
        n["polls"] += 1
        healthy_step(tmp_path, clk, n["polls"], n=1)
        if n["polls"] == 4:
            set_status(tmp_path, "DONE_PERIODIC")
        return real()

    sup.poll = counted
    assert sup.run() == 0
    assert n["polls"] == 4 and clk.slept == [300, 300, 300]


def test_second_supervisor_copy_refuses_to_start(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    holder = ps.add(9001, "python.exe", ts(T0))
    (tmp_path / "supervisor.lock").write_text(json.dumps({"pid": 9001, "image": "python.exe",
                                                          "create_time": ts(T0)}))
    sup = make_sup(tmp_path, ps, clk, my_pid=9002)
    assert sup.acquire_lock() is False
    holder.alive = False
    assert sup.acquire_lock() is True
    assert json.loads((tmp_path / "supervisor.lock").read_text())["pid"] == 9002


def test_dry_run_never_ends_or_relaunches_and_writes_no_state(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    real = make_sup(tmp_path, ps, clk)
    first_poll(real, tmp_path, clk)
    dry = make_sup(tmp_path, ps, clk, dry_run=True)
    (tmp_path / "fluent-0-error.log").write_text("Abnormal Exit!\n")
    clk.advance(minutes=5)
    assert dry.poll() == "continue"
    assert "DRY RUN" in (tmp_path / "supervisor_dryrun.log").read_text()
    st = state(tmp_path)
    st["phase"] = "ending"                                       # a real supervisor mid-end: the dry run still acts never
    (tmp_path / "supervisor_state.json").write_text(json.dumps(st))
    dry2 = make_sup(tmp_path, ps, clk, dry_run=True)
    clk.advance(minutes=5)
    assert dry2.poll() == "continue"
    assert ps.killed == [] and dry.popen.calls == [] and dry2.popen.calls == []
    assert state(tmp_path) == st                                 # the dry run wrote no state


def test_cli_dry_run_once_with_the_real_psutil_takes_no_action(tmp_path):
    set_status(tmp_path, "SOLVING")
    rc = S.main(["--design", "sine_cf2_020", "--run-dir", str(tmp_path), "--nproc", "16", "--cpus", "20-35",
                 "--until", "2099-01-01T00:00", "--once", "--dry-run"])
    assert rc == 0
    assert "verdict WAITING" in (tmp_path / "supervisor_dryrun.log").read_text()
    assert not (tmp_path / "supervisor_state.json").exists() and not (tmp_path / "supervisor.lock").exists()


# ---------------------------------------------------------------- the actual run-folder layout
def test_attempt2_progress_in_the_rewritten_pt_steps_csv_is_no_crash(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    sup = make_sup(tmp_path, ps, clk)
    first_poll(sup, tmp_path, clk)
    (tmp_path / "fluent-0-error.log").write_text("Abnormal Exit!\n")
    clk.advance(minutes=5)
    sup.poll()
    assert state(tmp_path)["attempt"] == 2
    (tmp_path / "pt_steps.csv").unlink()                         # attempt 2 rewrites pt_steps.csv (rows <= K) ...
    add_rows(tmp_path / "pt_steps.csv", 1, 2, clk.now())
    k = 3
    for _ in range(12):                                          # ... and appends to it: 60 min, no pt_steps_a2.csv
        clk.advance(minutes=5)
        add_rows(tmp_path / "pt_steps.csv", k, 1, clk.now())
        k += 1
        assert sup.poll() == "continue"
    assert not (tmp_path / S.A2_STEPS_NAME).exists()
    assert log(tmp_path).count("CRASHED") == 1 and len(sup.popen.calls) == 1


def test_the_save_used_is_the_resume_data_not_the_fallback(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    (tmp_path / "saves").mkdir()
    cyc = {"kind": "cycle", "step": 160, "flow_time_s": 0.8, "last_reset_t_s": 0.8, "sampled_time_s": 0.0,
           "path": str(tmp_path / "saves" / "d_c1.dat.h5"), "size": 5}
    roll = {"kind": "rolling", "step": 240, "flow_time_s": 1.2, "last_reset_t_s": 0.8, "sampled_time_s": 0.4,
            "path": str(tmp_path / "saves" / "d_r240.dat.h5"), "size": 5}
    (tmp_path / "saves" / "index.json").write_text(json.dumps({"case": "c.cas.h5", "saves": [cyc, roll]}))

    def stub(run_dir, design, nproc, cpus):
        return ["py.exe", "run", "--resume-data", roll["path"], "--resume-step", "240",
                "--fallback-data", cyc["path"], "--fallback-step", "160", "--attempt", "2"]

    sup = make_sup(tmp_path, ps, clk, relaunch_fn=stub)
    first_poll(sup, tmp_path, clk)
    (tmp_path / "fluent-0-error.log").write_text("Abnormal Exit!\n")
    clk.advance(minutes=5)
    sup.poll()
    r = state(tmp_path)["relaunch"]
    assert r["save"]["step"] == 240 and r["resumed_from_step"] == 240 and r["resumed_from_flow_time"] == 1.2


def test_hold_processes_exit_by_themselves_and_do_not_block_the_relaunch(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    holds = [ps.add(1100 + i, "python.exe", pr["run_script"]._ct + 2) for i in range(3)]
    for h in holds:
        h.dies_on_wait = True
    doc = json.loads((tmp_path / "pids.json").read_text())
    doc["processes"] += [rec("hold", h) for h in holds]
    (tmp_path / "pids.json").write_text(json.dumps(doc))
    sup = make_sup(tmp_path, ps, clk)
    first_poll(sup, tmp_path, clk)
    (tmp_path / "fluent-0-error.log").write_text("Abnormal Exit!\n")
    clk.advance(minutes=5)
    sup.poll()
    assert not any(h.pid in ps.killed for h in holds)            # never ended by the supervisor
    assert len(sup.popen.calls) == 1


def test_join_attempts_on_the_rewritten_pt_steps_csv_layout(tmp_path):
    _rows(tmp_path / "pt_steps_attempt1.csv", range(1, 251))     # the crashed attempt's own rows, kept
    lines = (tmp_path / "pt_steps_attempt1.csv").read_text().splitlines()[:241]       # header + steps 1..240 verbatim
    with (tmp_path / "pt_steps.csv").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
        for k in range(241, 301):                                # attempt 2 rows: other wall times
            fh.write(f"{k},{round(k * 0.005, 9)},0,2,12,False,55\n")
    rows, rerun = S.join_attempts(tmp_path)
    ft = [round(float(r["flow_time_s"]), 9) for r in rows]
    assert ft == [round(k * 0.005, 9) for k in range(1, 301)]
    assert {r["attempt"] for r in rows[:240]} == {1} and {r["attempt"] for r in rows[240:]} == {2}
    assert rerun == 10


def test_a_new_run_in_the_same_folder_starts_from_a_fresh_state(tmp_path):
    clk = FakeClock(T0)
    ps, pr, nodes = make_run(tmp_path, clk)
    sup = make_sup(tmp_path, ps, clk)
    first_poll(sup, tmp_path, clk)
    st = state(tmp_path)
    st["ended"], st["relaunch"] = "second crash", {"pid": 7001}            # an earlier run, over and done
    (tmp_path / "supervisor_state.json").write_text(json.dumps(st))
    clk.advance(minutes=60)
    ps2, pr2, nodes2 = make_run(tmp_path, clk)                  # relaunched by hand in the same folder
    sup2 = make_sup(tmp_path, ps2, clk)
    clk.advance(minutes=25)                                      # no row yet: still in the NEW run's grace
    assert sup2.poll() == "continue"
    st2 = state(tmp_path)
    assert st2["ended"] is None and st2["relaunch"] is None and st2["attempt"] == 1
    assert st2["attempts"]["1"]["launch"].startswith("2026-10-07T14:10")       # the new run script's start
    assert "new run in this folder" in log(tmp_path) and "in grace" in log(tmp_path).splitlines()[-1]
    assert list(tmp_path.glob("supervisor_state_prev_*.json"))


def test_default_relaunch_fn_uses_the_contract(monkeypatch, tmp_path):
    seen = []
    fake = types.ModuleType("pulsatile_top3_ops")
    fake.relaunch_command = lambda run_dir, design, nproc, cpus: seen.append((run_dir, design, nproc, cpus)) or ["x"]
    monkeypatch.setitem(sys.modules, "pulsatile_top3_ops", fake)
    assert S.default_relaunch_fn(tmp_path, "sine_v3_w1_04", 24, "8-31") == ["x"]
    assert seen == [(tmp_path, "sine_v3_w1_04", 24, "8-31")]


# ---------------------------------------------------------------- join_attempts (report files after a relaunch)
def _rows(path, steps, t0_step=None, dt_s=0.005):
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(HDR)
        for i, k in enumerate(steps):
            kk = k if t0_step is None else t0_step + i
            fh.write(f"{kk},{round(k * dt_s, 9)},0,1,10,False,60\n")


def test_join_attempts_drops_attempt1_rows_after_the_save_and_counts_reruns(tmp_path):
    _rows(tmp_path / "pt_steps.csv", range(1, 251))              # attempt 1 reached step 250 (1.25 s)
    _rows(tmp_path / "pt_steps_a2.csv", range(241, 301))         # attempt 2 resumed from the step-240 save
    (tmp_path / "supervisor_state.json").write_text(json.dumps(
        {"relaunch": {"save": {"step": 240, "flow_time": 1.2}, "resumed_from_step": 240}}))
    rows, rerun = S.join_attempts(tmp_path)
    ft = [float(r["flow_time_s"]) for r in rows]
    assert len(rows) == 300 and ft == sorted(ft)
    assert ft[0] == pytest.approx(0.005) and ft[-1] == pytest.approx(1.5)
    assert {r["attempt"] for r in rows[:240]} == {1} and {r["attempt"] for r in rows[240:]} == {2}
    assert rerun == 10                                           # steps 241-250 computed twice


def test_join_attempts_joins_by_flow_time_not_step_number(tmp_path):
    _rows(tmp_path / "pt_steps.csv", range(1, 245))
    _rows(tmp_path / "pt_steps_a2.csv", range(221, 260), t0_step=1)   # attempt 2 restarts its step column at 1
    rows, rerun = S.join_attempts(tmp_path)                      # no state: restored time = first a2 row - dt
    ft = [round(float(r["flow_time_s"]), 9) for r in rows]
    assert ft == [round(k * 0.005, 9) for k in range(1, 260)]
    assert rerun == 244 - 220
    rows2, rerun2 = S.join_attempts(tmp_path, restored_flow_time=1.1)
    assert len(rows2) == len(rows) and rerun2 == rerun


def test_join_attempts_without_a_relaunch_is_attempt1_unchanged(tmp_path):
    _rows(tmp_path / "pt_steps.csv", range(1, 11))
    rows, rerun = S.join_attempts(tmp_path)
    assert [int(r["step"]) for r in rows] == list(range(1, 11)) and rerun == 0
    assert {r["attempt"] for r in rows} == {1}
