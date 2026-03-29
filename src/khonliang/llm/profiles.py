"""
Model profiles — persistent performance data for scheduler tuning.

Stores per-model benchmarks (load time, inference time, VRAM, eviction
cost) in a YAML file. The scheduler reads this on startup for initial
scoring. As the system runs, actual stats refine the values. On shutdown,
updated profiles are saved back.

Usage:
    profiles = ModelProfiles("model_profiles.yaml")
    profiles.load()

    # Scheduler uses profiles for initial scoring
    scheduler = ModelScheduler(
        model_vram=profiles.get_vram_map(),
    )

    # After running, update profiles from actual stats
    profiles.update_from_stats(scheduler.get_all_stats())
    profiles.save()
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class ModelProfile:
    """Performance profile for a single model."""

    model: str
    vram_mb: int = 0
    avg_load_ms: float = 0.0
    avg_inference_ms: float = 0.0
    avg_evict_ms: float = 0.0
    max_tokens: int = 4096
    pin: bool = False  # never evict this model when loaded
    # Benchmark metadata
    benchmark_runs: int = 0
    last_benchmarked: Optional[str] = None
    # Runtime-updated stats (from actual usage)
    runtime_requests: int = 0
    runtime_avg_inference_ms: float = 0.0
    runtime_avg_load_ms: float = 0.0

    def effective_load_ms(self) -> float:
        """Best estimate of load time — prefer runtime data."""
        if self.runtime_avg_load_ms > 0:
            return self.runtime_avg_load_ms
        return self.avg_load_ms

    def effective_inference_ms(self) -> float:
        """Best estimate of inference time — prefer runtime data."""
        if self.runtime_avg_inference_ms > 0:
            return self.runtime_avg_inference_ms
        return self.avg_inference_ms

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v}


class ModelProfiles:
    """
    Persistent model performance profiles.

    Reads/writes a YAML file with per-model benchmark data.
    Falls back to a simple dict-based format if PyYAML is not installed.

    Example:
        profiles = ModelProfiles("model_profiles.yaml")
        profiles.load()
        profile = profiles.get("llama3.2:3b")
        if profile:
            print(f"Load time: {profile.effective_load_ms()}ms")
    """

    def __init__(self, path: str = "model_profiles.yaml"):
        self.path = Path(path)
        self._profiles: Dict[str, ModelProfile] = {}

    def load(self) -> int:
        """
        Load profiles from file. Returns number loaded.
        Creates default profiles if file doesn't exist.
        """
        if not self.path.exists():
            logger.debug(f"No profile file at {self.path}")
            return 0

        try:
            text = self.path.read_text()
            if HAS_YAML:
                data = yaml.safe_load(text) or {}
            else:
                import json
                data = json.loads(text)

            models = data.get("models", {})
            for model_name, values in models.items():
                values["model"] = model_name
                self._profiles[model_name] = ModelProfile(**values)

            logger.info(f"Loaded {len(self._profiles)} model profiles from {self.path}")
            return len(self._profiles)
        except Exception as e:
            logger.warning(f"Failed to load profiles from {self.path}: {e}")
            return 0

    def save(self) -> None:
        """Save current profiles to file."""
        data = {
            "models": {
                name: profile.to_dict()
                for name, profile in sorted(self._profiles.items())
            }
        }

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if HAS_YAML:
                self.path.write_text(yaml.dump(data, default_flow_style=False))
            else:
                import json
                self.path.write_text(json.dumps(data, indent=2))
            logger.info(f"Saved {len(self._profiles)} model profiles to {self.path}")
        except Exception as e:
            logger.warning(f"Failed to save profiles to {self.path}: {e}")

    def get(self, model: str) -> Optional[ModelProfile]:
        """Get profile for a model."""
        return self._profiles.get(model)

    def set(self, profile: ModelProfile) -> None:
        """Set or update a model profile."""
        self._profiles[profile.model] = profile

    def get_or_create(self, model: str) -> ModelProfile:
        """Get existing profile or create a blank one."""
        if model not in self._profiles:
            self._profiles[model] = ModelProfile(model=model)
        return self._profiles[model]

    def list_models(self) -> list:
        """List all profiled model names."""
        return list(self._profiles.keys())

    def get_pinned_models(self) -> list:
        """Get list of models marked as pinned (never evict)."""
        return [
            name for name, p in self._profiles.items() if p.pin
        ]

    # ------------------------------------------------------------------
    # Scheduler integration
    # ------------------------------------------------------------------

    def get_vram_map(self) -> Dict[str, int]:
        """Get model -> VRAM mapping for scheduler constructor."""
        return {
            name: p.vram_mb
            for name, p in self._profiles.items()
            if p.vram_mb > 0
        }

    def get_load_times(self) -> Dict[str, float]:
        """Get model -> avg load time in ms."""
        return {
            name: p.effective_load_ms()
            for name, p in self._profiles.items()
            if p.effective_load_ms() > 0
        }

    def get_inference_times(self) -> Dict[str, float]:
        """Get model -> avg inference time in ms."""
        return {
            name: p.effective_inference_ms()
            for name, p in self._profiles.items()
            if p.effective_inference_ms() > 0
        }

    def update_from_stats(self, model_stats: Dict[str, Any]) -> int:
        """
        Update profiles with runtime stats from the scheduler.

        Call this periodically or on shutdown to persist learned values.

        Args:
            model_stats: Dict from scheduler.get_status()["model_stats"]

        Returns:
            Number of profiles updated.
        """
        updated = 0
        for model, stats in model_stats.items():
            profile = self.get_or_create(model)
            avg_inf = stats.get("avg_inference_ms", 0)
            avg_load = stats.get("avg_load_ms", 0)
            total = stats.get("total_requests", 0)

            if avg_inf > 0:
                profile.runtime_avg_inference_ms = avg_inf
            if avg_load > 0:
                profile.runtime_avg_load_ms = avg_load
            if total > 0:
                profile.runtime_requests += total
            updated += 1

        return updated

    def seed_scheduler_stats(self, scheduler: Any) -> int:
        """
        Seed a scheduler's stats from profile data.

        Call on startup so the scheduler has initial estimates
        before any actual inference runs.

        Args:
            scheduler: ModelScheduler instance

        Returns:
            Number of models seeded.
        """
        seeded = 0
        for model, profile in self._profiles.items():
            stats = scheduler.get_stats(model)
            if profile.effective_inference_ms() > 0:
                stats.avg_inference_ms = profile.effective_inference_ms()
            if profile.effective_load_ms() > 0:
                stats.avg_load_ms = profile.effective_load_ms()
            seeded += 1
        return seeded
