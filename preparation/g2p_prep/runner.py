"""Shared per-scene parallel driver for the preparation-stage tools."""

import concurrent.futures
import time

from tqdm import tqdm


class SceneFailure(Exception):
    """Raised inside a worker; carries the scene id to the driver."""

    def __init__(self, scene, message):
        super().__init__(f"{scene}: {message}")
        self.scene = scene
        self.message = message


def format_seconds(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours:02d}:{minutes:02d}:{seconds % 60.0:06.3f}"


def run_scenes(worker, scenes, num_workers, desc):
    """Run ``worker(scene)`` over ``scenes``, collecting failures.

    Unlike the original scripts, exceptions are not swallowed: each failure is
    recorded and reported by the caller, which exits non-zero when any scene
    failed.

    Returns:
        ``(elapsed_seconds, skipped, failures)`` where ``failures`` is a list of
        ``(scene, message)``.
    """
    start = time.perf_counter()
    failures = []
    skipped = 0

    if num_workers <= 1:
        results = (_safe_call(worker, scene) for scene in scenes)
        results = tqdm(results, total=len(scenes), desc=desc)
        outcomes = list(results)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
            outcomes = list(
                tqdm(
                    pool.map(_SafeWorker(worker), scenes),
                    total=len(scenes),
                    desc=desc,
                )
            )

    for scene, status, message in outcomes:
        if status == "failed":
            failures.append((scene, message))
        elif status == "skipped":
            skipped += 1

    return time.perf_counter() - start, skipped, failures


class _SafeWorker:
    """Picklable wrapper so ``ProcessPoolExecutor`` can report per-scene errors."""

    def __init__(self, worker):
        self.worker = worker

    def __call__(self, scene):
        return _safe_call(self.worker, scene)


def _safe_call(worker, scene):
    try:
        status = worker(scene)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return scene, "failed", f"{type(exc).__name__}: {exc}"
    return scene, status or "done", ""


def report(split_timings, failures, continue_on_error):
    """Print a timing summary and return the process exit code."""
    print("\n=== Timing summary ===")
    total_time = 0.0
    total_scenes = 0
    for split, (elapsed, count, skipped) in split_timings.items():
        total_time += elapsed
        total_scenes += count
        avg = elapsed / count if count else 0.0
        print(
            f"{split}: {count} scenes ({skipped} skipped) in "
            f"{format_seconds(elapsed)} (avg {format_seconds(avg)}/scene)"
        )
    avg = total_time / total_scenes if total_scenes else 0.0
    print(
        f"total: {total_scenes} scenes in {format_seconds(total_time)} "
        f"(avg {format_seconds(avg)}/scene)"
    )

    if not failures:
        return 0

    print(f"\n=== {len(failures)} scene(s) failed ===")
    for scene, message in failures[:50]:
        print(f"  {scene}: {message}")
    if len(failures) > 50:
        print(f"  ... and {len(failures) - 50} more")

    if continue_on_error:
        print("\n--continue-on-error was set; exiting 0 despite failures.")
        return 0
    return 1
