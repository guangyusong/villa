import os
import json
import time
import click
import logging
import subprocess
import signal
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to track child processes
active_processes = []
executor = None


def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) and SIGTERM gracefully."""
    global active_processes, executor

    logger.info(f"\nReceived signal {signum}. Cleaning up...")

    # Terminate all active child processes
    for process in active_processes[:]:  # Use slice to copy list
        try:
            logger.info(f"Terminating process {process.pid}...")
            process.terminate()
            # Give process a moment to terminate gracefully
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                logger.info(f"Force killing process {process.pid}...")
                process.kill()
                process.wait()
        except Exception as e:
            logger.error(f"Error terminating process: {e}")

    # Shutdown the executor if it exists
    if executor:
        logger.info("Shutting down executor...")
        executor.shutdown(wait=False, cancel_futures=True)

    logger.info("Cleanup complete. Exiting.")
    sys.exit(0)


class JobTracker:
    """Manages tracking of completed and in-progress jobs."""

    def __init__(self, tracker_path: Path):
        self.tracker_path = tracker_path
        self.data = self._load_tracker()

    def _load_tracker(self) -> Dict:
        """Load existing tracker data or create new."""
        if self.tracker_path.exists():
            try:
                with open(self.tracker_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse tracker file {self.tracker_path}, creating new one")

        return {
            "completed": [],
            "failed": [],
            "in_progress": [],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_processed": 0,
                "total_successful": 0,
                "total_failed": 0
            }
        }

    def _save_tracker(self):
        """Save tracker data to disk."""
        self.data["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.tracker_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def is_completed(self, seed: Tuple[float, float, float]) -> bool:
        """Check if a seed has been completed."""
        seed_str = self._seed_to_str(seed)
        return seed_str in self.data["completed"]

    def is_failed(self, seed: Tuple[float, float, float]) -> bool:
        """Check if a seed has failed."""
        seed_str = self._seed_to_str(seed)
        return seed_str in self.data["failed"]

    def mark_in_progress(self, seed: Tuple[float, float, float]):
        """Mark a seed as in progress."""
        seed_str = self._seed_to_str(seed)
        if seed_str not in self.data["in_progress"]:
            self.data["in_progress"].append(seed_str)
            self._save_tracker()

    def mark_completed(self, seed: Tuple[float, float, float]):
        """Mark a seed as completed."""
        seed_str = self._seed_to_str(seed)

        # Remove from in_progress if present
        if seed_str in self.data["in_progress"]:
            self.data["in_progress"].remove(seed_str)

        # Remove from failed if it was previously failed
        if seed_str in self.data["failed"]:
            self.data["failed"].remove(seed_str)
            self.data["metadata"]["total_failed"] -= 1

        # Add to completed if not already there
        if seed_str not in self.data["completed"]:
            self.data["completed"].append(seed_str)
            self.data["metadata"]["total_successful"] += 1
            self.data["metadata"]["total_processed"] += 1

        self._save_tracker()

    def mark_failed(self, seed: Tuple[float, float, float]):
        """Mark a seed as failed."""
        seed_str = self._seed_to_str(seed)

        # Remove from in_progress if present
        if seed_str in self.data["in_progress"]:
            self.data["in_progress"].remove(seed_str)

        # Add to failed if not already there
        if seed_str not in self.data["failed"]:
            self.data["failed"].append(seed_str)
            self.data["metadata"]["total_failed"] += 1
            self.data["metadata"]["total_processed"] += 1

        self._save_tracker()

    def get_pending_seeds(self, all_seeds: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Get list of seeds that haven't been completed yet."""
        pending = []
        for seed in all_seeds:
            if not self.is_completed(seed):
                pending.append(seed)
        return pending

    def get_stats(self) -> Dict:
        """Get tracking statistics."""
        return {
            "completed": len(self.data["completed"]),
            "failed": len(self.data["failed"]),
            "in_progress": len(self.data["in_progress"]),
            "total_processed": self.data["metadata"]["total_processed"],
            "last_updated": self.data["metadata"]["last_updated"]
        }

    def reset_in_progress(self):
        """Reset all in-progress jobs (useful after a crash)."""
        count = len(self.data["in_progress"])
        self.data["in_progress"] = []
        self._save_tracker()
        return count

    def reset_failed(self):
        """Reset failed jobs to allow retry."""
        count = len(self.data["failed"])
        self.data["failed"] = []
        self.data["metadata"]["total_failed"] = 0
        self._save_tracker()
        return count

    @staticmethod
    def _seed_to_str(seed: Tuple[float, float, float]) -> str:
        """Convert seed tuple to string for storage."""
        return f"{int(seed[0])},{int(seed[1])},{int(seed[2])}"


class SeedRunner:
    """
    Seeding-only runner extracted from eval_surface_tracer.py with job tracking.

    - Loads config
    - Finds seed points either from a JSON file or by scanning an existing
      patches directory (explicit_seed only), with optional z-range filtering
    - Runs vc_grow_seg_from_seed in parallel for given seeds
    - Tracks completed jobs to allow restart
    - Optionally just reuses existing patches (use_existing_patches=true)
    """

    def __init__(self, config_path: str, reset_tracker: bool = False, retry_failed: bool = False):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.out_dir = Path(self.config["out_path"])  # required
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Initialize job tracker
        tracker_path = self.out_dir / "job_tracker.json"
        self.tracker = JobTracker(tracker_path)

        # Handle reset options
        if reset_tracker:
            logger.info("Resetting job tracker (all jobs will be re-run)")
            tracker_path.unlink(missing_ok=True)
            self.tracker = JobTracker(tracker_path)

        if retry_failed:
            count = self.tracker.reset_failed()
            logger.info(f"Reset {count} failed jobs for retry")

        # Reset any in-progress jobs (from previous interrupted run)
        in_progress_count = self.tracker.reset_in_progress()
        if in_progress_count > 0:
            logger.info(f"Reset {in_progress_count} in-progress jobs from previous run")

        # Hardcode patches directory to the specified path
        self.patches_dir = Path("/mnt/raid_nvme/volpkgs/PHerc172.volpkg/paths/")
        # Match behavior from eval_surface_tracer: if using existing patches,
        # allow directory to pre-exist. Otherwise, error if it exists to prevent
        # accidental reuse.
        self.patches_dir.mkdir(parents=True, exist_ok=self.config.get("use_existing_patches", False))

    def find_seed_points(self) -> List[Tuple[float, float, float]]:
        """
        Find seed points from either a JSON file (grouped by mode) or a folder
        of patches. Only seeds with mode "explicit_seed" are used. If z_range
        is provided, seeds are filtered to that range.
        """
        patches_path = Path(self.config["existing_patches_for_seeds"])  # required
        z_min, z_max = self.config.get("z_range", [-float("inf"), float("inf")])

        # Case 1: JSON file created by scripts/evaluation/get_seeds_from_paths.py
        if patches_path.is_file() and patches_path.suffix == ".json":
            with open(patches_path, 'r') as f:
                seeds_by_mode = json.load(f)
            seed_points = [
                (x, y, z)
                for (x, y, z) in seeds_by_mode.get("explicit_seed", [])
                if z_min <= z <= z_max
            ]
            logger.info(f"Loaded {len(seed_points)} seeds from JSON {patches_path}")
            # Optionally write a copy of the used seeds for reproducibility
            seeds_copy = self.out_dir / "seeds_used.json"
            with open(seeds_copy, 'w') as f:
                json.dump({"explicit_seed": seed_points}, f, indent=2)
            return seed_points

        # Case 2: Directory of patches with meta.json files
        if patches_path.is_dir():
            seed_points = []
            failed_count = 0
            for patch_dir in patches_path.iterdir():
                if not patch_dir.is_dir():
                    continue
                try:
                    meta_file = patch_dir / "meta.json"
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)
                    if meta.get("vc_gsfs_mode") != "explicit_seed":
                        continue
                    seed = meta.get("seed")
                    if not seed or len(seed) != 3:
                        continue
                    x, y, z = seed
                    if z_min <= z <= z_max:
                        seed_points.append((x, y, z))
                except Exception:
                    failed_count += 1
                    continue

            if failed_count:
                logger.warning(f"Failed to read meta.json from {failed_count} patches")
            logger.info(
                f"Found {len(seed_points)} explicit_seed seed points in z-range [{z_min}, {z_max}]"
            )

            # Also emit a seeds JSON for convenience, mirroring get_seeds_from_paths.py
            seeds_json = self.out_dir / "seeds_from_paths.json"
            with open(seeds_json, 'w') as f:
                json.dump({"explicit_seed": seed_points}, f, indent=2)
            logger.info(f"Wrote seed JSON to {seeds_json}")

            return seed_points

        logger.error(
            f"existing_patches_for_seeds path {patches_path} is neither a valid JSON file nor a directory"
        )
        return []

    def _run_vc_grow_seg_from_seed(self, params_file: Path, seed_point: Tuple[float, float, float]) -> bool:
        """Run vc_grow_seg_from_seed for a single seed point."""
        global active_processes

        # Check if already completed
        if self.tracker.is_completed(seed_point):
            logger.info(f"Skipping already completed seed: {seed_point}")
            return True

        # Mark as in progress
        self.tracker.mark_in_progress(seed_point)

        cmd = [
            f"{self.config['bin_path']}/vc_grow_seg_from_seed",
            self.config["surface_zarr_volume"],
            "/volpkgs/PHerc172.volpkg/paths/",
            str(params_file),
            str(int(seed_point[0])), str(int(seed_point[1])), str(int(seed_point[2]))
        ]

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"

        print(f"\n{'=' * 60}")
        print(f"STARTING SEED: {seed_point}")
        print(f"COMMAND: {' '.join(cmd)}")
        print(f"{'=' * 60}")

        # Use Popen instead of run to have better process control
        process = subprocess.Popen(cmd, env=env, text=True)
        active_processes.append(process)

        try:
            # Wait for process to complete
            returncode = process.wait()

            # Remove from active processes list
            if process in active_processes:
                active_processes.remove(process)

            print(f"\n{'=' * 60}")
            if returncode == 0:
                print(f"✓ FINISHED SEED: {seed_point} (SUCCESS)")
                print(f"{'=' * 60}\n")
                logger.info(f"Finished seeding run for seed {seed_point}")
                self.tracker.mark_completed(seed_point)
                return True
            else:
                print(f"✗ FINISHED SEED: {seed_point} (FAILED - return code: {returncode})")
                print(f"{'=' * 60}\n")
                logger.error(f"Failed seeding run for seed {seed_point} (return code: {returncode})")
                self.tracker.mark_failed(seed_point)
                return False
        except:
            # Remove from active processes list even if interrupted
            if process in active_processes:
                active_processes.remove(process)
            raise

    def run_seeding(self, seed_points: List[Tuple[float, float, float]]) -> None:
        """Run seeding for the given seed points, skipping already completed ones."""
        global executor

        # Get only pending seeds
        pending_seeds = self.tracker.get_pending_seeds(seed_points)

        # Print status
        stats = self.tracker.get_stats()
        logger.info(f"Job tracker status: {stats['completed']} completed, {stats['failed']} failed")
        logger.info(f"Found {len(pending_seeds)} pending seeds out of {len(seed_points)} total")

        if not pending_seeds:
            logger.info("All seeds have been processed! Nothing to do.")
            return

        logger.info(f"Running vc_grow_seg_from_seed for {len(pending_seeds)} pending seeds")

        seeding_params = self.config["vc_grow_seg_from_seed_params"]["seeding"].copy()
        seeding_params["mode"] = "seed"

        seeding_params_file = self.out_dir / "seeding_params.json"
        with open(seeding_params_file, 'w') as f:
            json.dump(seeding_params, f, indent=2)

        max_num_seeds = self.config.get("max_num_seeds", len(seed_points))
        parallel = int(self.config.get("seeding_parallel_processes", 1))

        # Apply max_num_seeds limit to pending seeds
        seeds_to_process = pending_seeds[:max_num_seeds]

        successful_runs = 0
        executor = ProcessPoolExecutor(max_workers=parallel)
        try:
            futures = []
            for seed_point in seeds_to_process:
                futures.append(
                    executor.submit(
                        self._run_vc_grow_seg_from_seed,
                        seeding_params_file,
                        seed_point,
                    )
                )

            for future in as_completed(futures):
                try:
                    if future.result():
                        successful_runs += 1
                except Exception as e:
                    logger.error(f"Error in seed growth: {e}")
        finally:
            executor.shutdown(wait=False)
            executor = None

        # Final statistics
        final_stats = self.tracker.get_stats()
        logger.info(
            f"Session complete: {successful_runs} successful runs in this session. "
            f"Total stats: {final_stats['completed']} completed, {final_stats['failed']} failed"
        )

        # Summarize created patches
        created_patches = []
        for patch_dir in self.patches_dir.iterdir():
            if patch_dir.is_dir() and (patch_dir / "meta.json").exists():
                created_patches.append(patch_dir)
        logger.info(f"Found {len(created_patches)} total patches in {self.patches_dir}")

    def run(self) -> None:
        try:
            if self.config.get("use_existing_patches", False):
                existing_patches = [
                    p for p in self.patches_dir.iterdir()
                    if p.is_dir() and (p / "meta.json").exists()
                ]
                logger.info(
                    f"use_existing_patches=true; skipping seeding. Found {len(existing_patches)} patches in {self.patches_dir}"
                )
                return

            # Otherwise, find seeds then run seeding
            seed_points = self.find_seed_points()
            if not seed_points:
                raise RuntimeError("No seed points found")
            self.run_seeding(seed_points)

        except Exception as e:
            logger.error(f"Error: {e}")
            raise


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--reset-tracker', is_flag=True, help='Reset the job tracker and re-run all jobs')
@click.option('--retry-failed', is_flag=True, help='Retry previously failed jobs')
@click.option('--show-stats', is_flag=True, help='Show job tracker statistics and exit')
def main(config_file: str, reset_tracker: bool, retry_failed: bool, show_stats: bool):
    """
    Run seed processing with job tracking for restart capability.

    The job tracker maintains a record of completed, failed, and in-progress jobs
    in job_tracker.json in the output directory. This allows the script to be
    restarted and resume from where it left off.

    Examples:
        # Normal run (will skip already completed seeds)
        python script.py config.json

        # Reset and re-run everything
        python script.py config.json --reset-tracker

        # Retry failed jobs
        python script.py config.json --retry-failed

        # Just show statistics
        python script.py config.json --show-stats
    """

    if show_stats:
        # Just show stats and exit
        with open(config_file, 'r') as f:
            config = json.load(f)
        out_dir = Path(config["out_path"])
        tracker_path = out_dir / "job_tracker.json"

        if not tracker_path.exists():
            print("No job tracker found. No jobs have been run yet.")
            return

        tracker = JobTracker(tracker_path)
        stats = tracker.get_stats()
        print("\nJob Tracker Statistics:")
        print(f"  Completed:     {stats['completed']}")
        print(f"  Failed:        {stats['failed']}")
        print(f"  In Progress:   {stats['in_progress']}")
        print(f"  Total Processed: {stats['total_processed']}")
        print(f"  Last Updated:  {stats['last_updated']}")
        return

    runner = SeedRunner(config_file, reset_tracker=reset_tracker, retry_failed=retry_failed)
    runner.run()


if __name__ == '__main__':
    main()