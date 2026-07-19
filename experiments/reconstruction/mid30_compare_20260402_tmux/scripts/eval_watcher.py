#!/usr/bin/env python3
import json
import os
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "experiments" / "reconstruction" / "mid30_compare_20260402_tmux"
SELECTED = [
    line.strip()
    for line in (ROOT / "selected_samples.txt").read_text().splitlines()
    if line.strip()
]
ORIG_ROOT = Path(
    os.environ.get(
        "MESH_ROOT",
        str(REPO_ROOT / "instruction_guidance" / "r1.0.1" / "reconstruction"),
    )
)
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python")
LOG_PATH = ROOT / "logs" / "eval_watcher.log"
SUMMARY_PATH = ROOT / "eval_summary.json"
PROGRESS_PATH = ROOT / "progress.json"


def log(message: str) -> None:
    with LOG_PATH.open("a") as fh:
        fh.write(f"[{time.strftime('%F %T')}] {message}\n")


def run_eval(method: str, sample: str, quad_path: Path, out_json: Path) -> bool:
    cmd = [
        PYTHON_BIN,
        "-m",
        "eval.evaluate",
        "--quad_mesh",
        str(quad_path),
        "--orig_mesh",
        str(ORIG_ROOT / f"{sample}.obj"),
        "--output_json",
        str(out_json),
    ]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    log(f"EVAL {method} {sample}")
    with LOG_PATH.open("a") as fh:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=fh, stderr=fh, env=env)
    log(f"EXIT {method} {sample} rc={proc.returncode}")
    return proc.returncode == 0


def main() -> None:
    ROOT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    while True:
        all_done = True
        done_pairs = 0
        total_pairs = len(SELECTED) * 2
        summary = {}
        for sample in SELECTED:
            sample_entry = {}
            pairs = [
                ("ours", ROOT / "ours" / sample / "quad_meshes" / f"{sample}_quad.obj", ROOT / f"eval_ours_{sample}.json"),
                ("baseline", ROOT / "baseline_runs" / sample / "quad_meshes" / f"{sample}_quad.obj", ROOT / f"eval_baseline_{sample}.json"),
            ]
            for method, quad_path, out_json in pairs:
                if out_json.exists():
                    try:
                        sample_entry[method] = json.loads(out_json.read_text())
                    except Exception:
                        sample_entry[method] = {"error": "failed_to_parse"}
                    done_pairs += 1
                    continue
                if quad_path.exists():
                    ok = run_eval(method, sample, quad_path, out_json)
                    if ok and out_json.exists():
                        sample_entry[method] = json.loads(out_json.read_text())
                    else:
                        sample_entry[method] = {"error": "eval_failed"}
                    done_pairs += 1
                else:
                    sample_entry[method] = {"status": "pending"}
                    all_done = False
            summary[sample] = sample_entry

        SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        PROGRESS_PATH.write_text(
            json.dumps(
                {
                    "done_pairs": done_pairs,
                    "total_pairs": total_pairs,
                    "completed_ratio": (done_pairs / total_pairs) if total_pairs else 1.0,
                },
                indent=2,
            )
        )
        if all_done:
            break
        time.sleep(45)


if __name__ == "__main__":
    main()
