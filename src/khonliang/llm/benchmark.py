"""
LLM benchmark — measures model performance for scheduler tuning.

Runs each model through standard tests to measure:
- Cold load time (first request after model is not loaded)
- Warm inference time (subsequent requests)
- VRAM usage (via Ollama API)
- Token throughput

Results are saved to a model_profiles.yaml file that the scheduler
reads on startup.

Usage:
    # Benchmark all available models
    python -m khonliang.llm.benchmark

    # Benchmark specific models
    python -m khonliang.llm.benchmark --models llama3.2:3b qwen2.5:7b

    # Custom output path
    python -m khonliang.llm.benchmark --output profiles.yaml

    # Quick mode (fewer iterations)
    python -m khonliang.llm.benchmark --quick

    # Validate models are loadable
    python -m khonliang.llm.benchmark --validate
"""

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

from khonliang.llm.profiles import ModelProfiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Standard benchmark prompts — short, medium, long
BENCHMARK_PROMPTS = {
    "short": "What is 2 + 2?",
    "medium": (
        "Explain the difference between a stack and a queue "
        "in computer science. Be concise."
    ),
    "long": (
        "Write a detailed comparison of three sorting algorithms: "
        "quicksort, mergesort, and heapsort. Include time complexity, "
        "space complexity, stability, and best use cases for each."
    ),
}

WARMUP_PROMPT = "Hi"


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single model."""

    model: str
    available: bool = False
    cold_load_ms: float = 0.0
    warm_inference_ms: List[float] = field(default_factory=list)
    avg_inference_ms: float = 0.0
    p50_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    tokens_per_second: float = 0.0
    vram_mb: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "available": self.available,
            "cold_load_ms": round(self.cold_load_ms),
            "avg_inference_ms": round(self.avg_inference_ms),
            "p50_inference_ms": round(self.p50_inference_ms),
            "p95_inference_ms": round(self.p95_inference_ms),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "vram_mb": self.vram_mb,
            "runs": len(self.warm_inference_ms),
            "error": self.error,
        }


class ModelBenchmark:
    """
    Benchmarks Ollama models for scheduler tuning.

    Example:
        bench = ModelBenchmark(ollama_url="http://localhost:11434")
        results = await bench.run_all()
        bench.save_profiles(results, "model_profiles.yaml")
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        iterations: int = 3,
        max_tokens: int = 200,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.iterations = iterations
        self.max_tokens = max_tokens

    async def list_models(self) -> List[str]:
        """Get available models from Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    data = await resp.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Cannot reach Ollama at {self.ollama_url}: {e}")
            return []

    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get model details including size."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/show",
                    json={"name": model},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            logger.debug(f"Could not get model info for {model}: {e}")
        return {}

    async def unload_model(self, model: str) -> bool:
        """Unload a model from Ollama to measure cold load."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": model, "keep_alive": 0},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def generate(
        self, model: str, prompt: str, max_tokens: int = 200
    ) -> Dict[str, Any]:
        """Run a single generation and return timing data."""
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": 0.1,
                        },
                    },
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    data = await resp.json()
                    elapsed_ms = (time.time() - start) * 1000

                    return {
                        "elapsed_ms": elapsed_ms,
                        "eval_count": data.get("eval_count", 0),
                        "eval_duration_ns": data.get("eval_duration", 0),
                        "total_duration_ns": data.get("total_duration", 0),
                        "prompt_eval_count": data.get("prompt_eval_count", 0),
                        "success": True,
                    }
        except Exception as e:
            return {
                "elapsed_ms": (time.time() - start) * 1000,
                "success": False,
                "error": str(e),
            }

    async def benchmark_model(self, model: str) -> BenchmarkResult:
        """
        Full benchmark of a single model.

        Steps:
        1. Unload model (cold start)
        2. First request = cold load time
        3. N warm requests = inference time
        4. Collect VRAM usage
        """
        result = BenchmarkResult(model=model)
        logger.info(f"Benchmarking {model}...")

        # Step 1: Unload for cold start measurement
        logger.info(f"  Unloading {model}...")
        await self.unload_model(model)
        await asyncio.sleep(1)

        # Step 2: Cold load (first request)
        logger.info("  Cold load test...")
        cold = await self.generate(model, WARMUP_PROMPT, max_tokens=1)
        if not cold.get("success"):
            result.error = cold.get("error", "Cold load failed")
            logger.error(f"  FAILED: {result.error}")
            return result

        result.available = True
        result.cold_load_ms = cold["elapsed_ms"]
        logger.info(f"  Cold load: {result.cold_load_ms:.0f}ms")

        # Step 3: Warm inference runs
        logger.info(f"  Running {self.iterations} warm inference tests...")
        total_tokens = 0
        total_eval_ns = 0

        for i, (name, prompt) in enumerate(BENCHMARK_PROMPTS.items()):
            for run in range(self.iterations):
                gen = await self.generate(model, prompt, self.max_tokens)
                if gen.get("success"):
                    result.warm_inference_ms.append(gen["elapsed_ms"])
                    total_tokens += gen.get("eval_count", 0)
                    total_eval_ns += gen.get("eval_duration_ns", 0)
                    logger.info(
                        f"    {name} run {run + 1}: "
                        f"{gen['elapsed_ms']:.0f}ms, "
                        f"{gen.get('eval_count', 0)} tokens"
                    )

        # Calculate stats
        if result.warm_inference_ms:
            sorted_times = sorted(result.warm_inference_ms)
            result.avg_inference_ms = (
                sum(sorted_times) / len(sorted_times)
            )
            result.p50_inference_ms = sorted_times[len(sorted_times) // 2]
            p95_idx = int(len(sorted_times) * 0.95)
            result.p95_inference_ms = sorted_times[min(p95_idx, len(sorted_times) - 1)]

            if total_eval_ns > 0:
                result.tokens_per_second = (
                    total_tokens / (total_eval_ns / 1e9)
                )

        # Step 4: VRAM estimate from model info
        info = await self.get_model_info(model)
        if info:
            params = info.get("details", {}).get("parameter_size", "")
            if "B" in params:
                try:
                    param_b = float(params.replace("B", ""))
                    # Rough: 1B params ≈ 600MB VRAM at Q4 quantization
                    result.vram_mb = int(param_b * 600)
                except ValueError:
                    logger.debug(f"Could not parse parameter size: {params}")

        logger.info(
            f"  Result: avg={result.avg_inference_ms:.0f}ms, "
            f"p50={result.p50_inference_ms:.0f}ms, "
            f"tps={result.tokens_per_second:.1f}, "
            f"cold={result.cold_load_ms:.0f}ms"
        )

        return result

    async def validate_model(self, model: str) -> Dict[str, Any]:
        """Quick validation — just check if model loads and responds."""
        logger.info(f"Validating {model}...")
        gen = await self.generate(model, "Say hello.", max_tokens=10)
        success = gen.get("success", False)
        status = "OK" if success else f"FAILED: {gen.get('error', '?')}"
        logger.info(f"  {model}: {status}")
        return {
            "model": model,
            "valid": success,
            "response_ms": gen.get("elapsed_ms", 0),
            "error": gen.get("error"),
        }

    async def run_all(
        self, models: Optional[List[str]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark all specified models (or all available)."""
        if models is None:
            models = await self.list_models()
            if not models:
                logger.error("No models available")
                return []

        logger.info(f"Benchmarking {len(models)} models: {models}")
        results = []
        for model in models:
            result = await self.benchmark_model(model)
            results.append(result)
            logger.info("")

        return results

    async def validate_all(
        self, models: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Quick validation of all models."""
        if models is None:
            models = await self.list_models()

        results = []
        for model in models:
            results.append(await self.validate_model(model))

        return results

    def save_profiles(
        self,
        results: List[BenchmarkResult],
        output_path: str = "model_profiles.yaml",
    ) -> None:
        """Save benchmark results as model profiles."""
        profiles = ModelProfiles(output_path)
        # Load existing to preserve runtime stats
        profiles.load()

        for result in results:
            if not result.available:
                continue

            profile = profiles.get_or_create(result.model)
            profile.avg_load_ms = result.cold_load_ms
            profile.avg_inference_ms = result.avg_inference_ms
            profile.benchmark_runs = len(result.warm_inference_ms)
            if result.vram_mb > 0:
                profile.vram_mb = result.vram_mb

            import datetime

            profile.last_benchmarked = datetime.datetime.now().isoformat()
            profiles.set(profile)

        profiles.save()
        logger.info(f"Saved profiles to {output_path}")

    def print_results(self, results: List[BenchmarkResult]) -> None:
        """Print a formatted results table."""
        print(f"\n{'Model':<25} {'Load':>8} {'Avg':>8} {'P50':>8} "
              f"{'P95':>8} {'TPS':>8} {'VRAM':>8} {'Status':<8}")
        print("-" * 90)

        for r in results:
            if r.available:
                print(
                    f"{r.model:<25} "
                    f"{r.cold_load_ms:>7.0f}ms "
                    f"{r.avg_inference_ms:>7.0f}ms "
                    f"{r.p50_inference_ms:>7.0f}ms "
                    f"{r.p95_inference_ms:>7.0f}ms "
                    f"{r.tokens_per_second:>7.1f} "
                    f"{r.vram_mb:>7}MB "
                    f"{'OK':<8}"
                )
            else:
                print(f"{r.model:<25} {'':>8} {'':>8} {'':>8} "
                      f"{'':>8} {'':>8} {'':>8} FAILED")


async def run_benchmark(args):
    """Main benchmark entry point."""
    bench = ModelBenchmark(
        ollama_url=args.ollama_url,
        iterations=1 if args.quick else 3,
        max_tokens=50 if args.quick else 200,
    )

    models = args.models if args.models else None

    if args.validate:
        results = await bench.validate_all(models)
        print(f"\n{'Model':<25} {'Status':<10} {'Response':>10}")
        print("-" * 50)
        for r in results:
            status = "OK" if r["valid"] else "FAILED"
            ms = f"{r['response_ms']:.0f}ms" if r["valid"] else r.get("error", "")[:20]
            print(f"{r['model']:<25} {status:<10} {ms:>10}")
        return

    results = await bench.run_all(models)
    bench.print_results(results)
    bench.save_profiles(results, args.output)


def main():
    parser = argparse.ArgumentParser(
        prog="khonliang.llm.benchmark",
        description="Benchmark Ollama models for scheduler tuning",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models to benchmark (default: all available)",
    )
    parser.add_argument(
        "--output", default="model_profiles.yaml",
        help="Output profiles file path",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer iterations, shorter prompts",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate only: check models load and respond",
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
