"""Benchmark speed and peak memory of the DS classifier pipeline on large synthetic data.

Generates synthetic inputs (default 6,000,000 address records) in a separate process, then runs the
same steps as dscm_run.ipynb with every training_functions step wrapped in a timer and a background
RSS sampler. Results are printed and written to data/output/benchmarks/.

Examples:
    .venv\\Scripts\\python scripts\\benchmark_pipeline.py
    .venv\\Scripts\\python scripts\\benchmark_pipeline.py --records 500000 --num-rounds 200
    .venv\\Scripts\\python scripts\\benchmark_pipeline.py --memory-limit-gb 12 --regenerate
"""
import argparse
import functools
import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

os.environ.setdefault("MPLBACKEND", "Agg")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from functions import training_functions as tf  # noqa: E402

CLASS_NAMES = ['OTHER', 'x11', 'x21', 'x22', 'x31', 'x32', 'x33', 'x34', 'x35', 'x91', 'x92', 'x93', 'x94']
# training_functions steps to time; nested calls (e.g. loaders inside the workflow) are reported indented
INSTRUMENTED = [
    'generate_training_data_workflow',
    'load_ar_df',
    'load_cad_gdf',
    'load_bld_gdf',
    'generate_ar_cad_linkage',
    'generate_bld_cad_linkage',
    'generate_ar_building_attachment',
    'train_test_model',
    '_load_model_frame',
    'complete_model_evaluation',
    'model_test_predictions',
    'predict_candidates',
]
GB = 1024 ** 3


class StageProfiler:
    """Times named stages and tracks peak RSS within each one using a sampling thread."""

    def __init__(self, memory_limit_bytes, interval=0.05, on_limit=None):
        self.process = psutil.Process()
        self.memory_limit = memory_limit_bytes
        self.interval = interval
        self.on_limit = on_limit
        self.stages = []
        self._active = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.overall_peak = self.process.memory_info().rss
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def _sample(self):
        while not self._stop.wait(self.interval):
            rss = self.process.memory_info().rss
            with self._lock:
                self.overall_peak = max(self.overall_peak, rss)
                for stage in self._active:
                    stage['peak_rss'] = max(stage['peak_rss'], rss)
            if rss > self.memory_limit and self.on_limit:
                self.on_limit(rss)
                return

    def wrap(self, name, func):
        @functools.wraps(func)
        def timed(*args, **kwargs):
            rss = self.process.memory_info().rss
            stage = {'name': name, 'depth': len(self._active), 'start_rss': rss, 'peak_rss': rss}
            with self._lock:
                self.stages.append(stage)
                self._active.append(stage)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                stage['seconds'] = time.perf_counter() - start
                stage['end_rss'] = self.process.memory_info().rss
                with self._lock:
                    stage['peak_rss'] = max(stage['peak_rss'], stage['end_rss'])
                    self._active.remove(stage)
        return timed


def input_paths(period, state):
    base = tf.LOCAL_INPUT_DIR
    return [
        base / f"{period}_dsim" / f"{period}_all_res_{state}.parquet",
        base / f"{period}_cad" / f"{period}_cad_{state}.parquet",
        base / f"{period}_bld" / f"{period}_bld_{state}.parquet",
    ]


def parquet_rows(path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


def generate_inputs(args):
    """Run the generator as a child process so its memory is not counted against the pipeline."""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_synthetic_data.py"),
           "--period", args.period, "--state", args.state, "--records", str(args.records), "--seed", str(args.seed)]
    print(f"Generating {args.records:,} synthetic records...")
    start = time.perf_counter()
    child = psutil.Popen(cmd)
    peak = 0
    while child.poll() is None:
        # The venv python.exe is a launcher, so the real interpreter is a child process
        rss = 0
        try:
            for proc in [child, *child.children(recursive=True)]:
                rss += proc.memory_info().rss
        except psutil.NoSuchProcess:
            pass
        peak = max(peak, rss)
        time.sleep(0.2)
    if child.returncode != 0:
        sys.exit(f"Synthetic data generation failed with exit code {child.returncode}")
    return {'seconds': time.perf_counter() - start, 'peak_rss': peak}


def format_table(stages):
    lines = [
        f"{'Stage':<46} {'Time (s)':>10} {'Peak RSS (GB)':>14} {'Peak increase (GB)':>19} {'End RSS (GB)':>13}",
        "-" * 106,
    ]
    for s in stages:
        name = "  " * s['depth'] + s['name']
        seconds = f"{s['seconds']:.1f}" if 'seconds' in s else "running"
        end = f"{s['end_rss'] / GB:.2f}" if 'end_rss' in s else "-"
        lines.append(
            f"{name:<46} {seconds:>10} {s['peak_rss'] / GB:>14.2f} "
            f"{(s['peak_rss'] - s['start_rss']) / GB:>19.2f} {end:>13}"
        )
    return "\n".join(lines)


def write_report(report, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"benchmark_{report['config']['period']}_{report['config']['state']}_{report['config']['records']}"
    (report_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, default=str))
    (report_dir / f"{stem}.txt").write_text(report['summary'])
    print(f"\nReport written to {report_dir / stem}.json / .txt")


def build_summary(report):
    cfg, sysinfo = report['config'], report['system']
    lines = [
        f"DS classifier benchmark - {report['status']}",
        f"Run at {report['timestamp']}",
        f"Records: {cfg['records']:,} requested | inputs: {report.get('input_rows', {})}",
        f"Config: state={cfg['state']} pool={cfg['pool']} method={cfg['method']} "
        f"extended_features={cfg['extended_features']} num_rounds={cfg['num_rounds']}",
        f"System: {sysinfo['cpu']} | {sysinfo['logical_cpus']} threads | "
        f"{sysinfo['total_ram_gb']:.1f} GB RAM ({sysinfo['available_ram_gb_at_start']:.1f} GB free at start)",
        "",
    ]
    if report.get('generation'):
        g = report['generation']
        lines.append(f"Synthetic data generation: {g['seconds']:.1f} s, peak {g['peak_rss'] / GB:.2f} GB (separate process)\n")
    lines.append(format_table(report['stages']))
    lines += [
        "",
        f"Total pipeline time: {report.get('total_seconds', float('nan')):.1f} s",
        f"Overall peak RSS: {report['overall_peak_rss'] / GB:.2f} GB "
        f"(baseline after imports {report['baseline_rss'] / GB:.2f} GB)",
    ]
    if report.get('output_rows'):
        lines.append(f"Output rows: {report['output_rows']}")
    if report.get('validation'):
        lines.append(f"Validation: {report['validation']}")
    if report.get('error'):
        lines.append(f"\nERROR: {report['error']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=int, default=6_000_000)
    parser.add_argument("--period", default=None, help="defaults to perf<records>, kept apart from the a25 test data")
    parser.add_argument("--state", default="SA")
    parser.add_argument("--pool", default="all")
    parser.add_argument("--method", default="building_target")
    parser.add_argument("--extended-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adhoc-output", help="folder (relative to data/output, or absolute) to write all outputs to instead of the quarter folders")
    parser.add_argument("--collapse-other-classes", action=argparse.BooleanOptionalAction, default=True,
                        help="merge x35/x91-x94 into OTHER before training (--no-collapse-other-classes keeps them)")
    parser.add_argument("--num-rounds", type=int, help="cap boosting rounds (default: the state's configured rounds)")
    parser.add_argument("--regenerate", action="store_true", help="rebuild synthetic inputs even if they exist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-limit-gb", type=float,
                        help="abort if RSS exceeds this (default: 85%% of physical RAM)")
    args = parser.parse_args()
    args.period = args.period or f"perf{args.records}"

    vm = psutil.virtual_memory()
    memory_limit = (args.memory_limit_gb * GB) if args.memory_limit_gb else vm.total * 0.85
    report = {
        'status': 'running',
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'config': {**vars(args), 'num_rounds': args.num_rounds or tf._STATE_NUM_ROUNDS[args.state],
                   'memory_limit_gb': memory_limit / GB},
        'system': {
            'cpu': platform.processor(),
            'logical_cpus': psutil.cpu_count(),
            'total_ram_gb': vm.total / GB,
            'available_ram_gb_at_start': vm.available / GB,
            'python': platform.python_version(),
        },
        'stages': [],
    }
    report_dir = tf.LOCAL_OUTPUT_DIR / "benchmarks"

    if args.regenerate or not all(p.exists() for p in input_paths(args.period, args.state)):
        report['generation'] = generate_inputs(args)
    report['input_rows'] = {p.parent.name: parquet_rows(p) for p in input_paths(args.period, args.state)}
    print(f"Input rows: {report['input_rows']}")

    if args.num_rounds:
        tf._STATE_NUM_ROUNDS[args.state] = args.num_rounds

    def abort(rss):
        report['status'] = 'aborted - memory limit'
        report['error'] = f"RSS reached {rss / GB:.2f} GB, above the {memory_limit / GB:.2f} GB limit"
        report['overall_peak_rss'] = profiler.overall_peak
        report['summary'] = build_summary(report)
        print("\n" + report['summary'])
        write_report(report, report_dir)
        os._exit(3)

    profiler = StageProfiler(memory_limit, on_limit=abort)
    report['stages'] = profiler.stages
    report['baseline_rss'] = profiler.process.memory_info().rss
    for name in INSTRUMENTED:
        setattr(tf, name, profiler.wrap(name, getattr(tf, name)))

    start = time.perf_counter()
    try:
        with profiler:
            tf.set_output_location(args.adhoc_output)
            tf.start_log(args.state)
            tf.generate_training_data_workflow(args.period, args.state, args.pool, args.method, args.extended_features)
            model, X_val, y_val, le, class_set = tf.train_test_model(
                args.period, args.state, args.method, args.collapse_other_classes
            )
            class_names = [name for name in CLASS_NAMES if name in class_set]
            evaluation = tf.complete_model_evaluation(
                args.period, args.state, args.method, model,
                X_val.drop(columns='ARID'), y_val[['DWELLING_STRUCT']], class_names=class_names
            )
            tf.model_test_predictions(args.period, args.state, args.method, evaluation, X_val, y_val,
                                      class_names=class_names, le=le)
            tf.predict_candidates(args.period, args.state, args.method, model, class_set=class_names, pool=args.pool)
        report['status'] = 'completed'
        report['validation'] = {
            'rows': len(X_val),
            'boosting_rounds_used': model.num_boosted_rounds(),
            'seconds_per_round': round(
                next(s['seconds'] for s in profiler.stages if s['name'] == 'train_test_model') / model.num_boosted_rounds(), 3
            ),
            'accuracy': round(float(evaluation['metrics']['accuracy']), 4),
            'f1_macro': round(float(evaluation['metrics']['f1_macro']), 4),
        }
        report['output_rows'] = {
            'training_table': parquet_rows(tf._training_output_path(args.period, args.state, args.pool)),
        }
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report['total_seconds'] = time.perf_counter() - start
        report['overall_peak_rss'] = max(profiler.overall_peak, getattr(profiler.process.memory_info(), 'peak_wset', 0))
        report['summary'] = build_summary(report)
        print("\n" + report['summary'])
        write_report(report, report_dir)


if __name__ == "__main__":
    main()
