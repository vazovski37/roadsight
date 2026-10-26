"""Run a queue of experiments on one GPU, several at a time, resumably.

Each job is {name, config, set: [overrides], after: [job names]} or {name, cmd: "..."} for an
arbitrary command (e.g. an evaluation). A job is skipped when runs/<name>/metrics.json (or
results/<name>.json for commands) already exists, so the queue can be restarted after a crash.

    python tools/run_queue.py queue.yaml --slots 2
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml


def done(job):
    if "cmd" in job:
        return Path("results", f"{job['name']}.json").exists()
    return Path("runs", job["name"], "metrics.json").exists()


def command(job):
    """argv list (no shell, so quoting behaves the same on Linux and Windows)."""
    if "cmd" in job:
        return shlex.split(job["cmd"])
    return [sys.executable, "tools/train.py", job["config"], f"name={job['name']}", *job.get("set", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--slots", type=int, default=2)
    args = ap.parse_args()
    jobs = yaml.safe_load(Path(args.queue).read_text())
    Path("logs").mkdir(exist_ok=True)
    running, failed = {}, set()
    while True:
        for name, (proc, start) in list(running.items()):
            if proc.poll() is not None:
                ok = proc.returncode == 0
                print(f"{time.strftime('%H:%M:%S')} {'DONE' if ok else 'FAILED'} {name} "
                      f"({(time.time() - start) / 60:.1f} min)", flush=True)
                if not ok:
                    failed.add(name)
                del running[name]
        pending = [j for j in jobs if not done(j) and j["name"] not in running and j["name"] not in failed]
        # a job whose dependency failed can never run
        for j in pending:
            if any(a in failed for a in j.get("after", [])):
                failed.add(j["name"])
                print(f"SKIP {j['name']} (dependency failed)", flush=True)
        pending = [j for j in pending if j["name"] not in failed]
        if not pending and not running:
            break
        for j in pending:
            if len(running) >= args.slots:
                break
            if all(done(next(x for x in jobs if x["name"] == a)) for a in j.get("after", [])):
                log = open(Path("logs", f"{j['name']}.log"), "w")
                print(f"{time.strftime('%H:%M:%S')} START {j['name']}: {' '.join(command(j))}", flush=True)
                proc = subprocess.Popen(command(j), stdout=log, stderr=subprocess.STDOUT, env=os.environ)
                running[j["name"]] = (proc, time.time())
        time.sleep(10)
    print("QUEUE FINISHED", "failed: " + ", ".join(sorted(failed)) if failed else "all ok", flush=True)


if __name__ == "__main__":
    main()
