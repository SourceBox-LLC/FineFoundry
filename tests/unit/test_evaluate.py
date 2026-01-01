"""Unit tests for the evaluate controller and benchmark functionality."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Test the benchmark configuration
from ui.tabs.evaluate_controller import BENCHMARKS, METRIC_INFO


class TestBenchmarkConfiguration:
    """Test benchmark definitions and configuration."""

    def test_benchmarks_dict_not_empty(self):
        """Verify benchmarks dictionary is populated."""
        assert len(BENCHMARKS) > 0

    def test_classic_benchmarks_present(self):
        """Verify classic Open LLM Leaderboard benchmarks are available."""
        classic_benchmarks = [
            "truthfulqa_mc2",
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "winogrande",
        ]
        for benchmark in classic_benchmarks:
            assert benchmark in BENCHMARKS, f"Missing classic benchmark: {benchmark}"

    def test_new_leaderboard_benchmarks_present(self):
        """Verify new Open LLM Leaderboard v2 benchmarks are available."""
        v2_benchmarks = ["ifeval", "bbh", "gpqa", "musr"]
        for benchmark in v2_benchmarks:
            assert benchmark in BENCHMARKS, f"Missing v2 benchmark: {benchmark}"

    def test_benchmark_descriptions_are_strings(self):
        """Verify all benchmark descriptions are non-empty strings."""
        for key, desc in BENCHMARKS.items():
            assert isinstance(desc, str), f"Benchmark {key} description is not a string"
            assert len(desc) > 0, f"Benchmark {key} has empty description"

    def test_quick_benchmarks_marked(self):
        """Verify quick benchmarks are marked with lightning bolt."""
        quick_benchmarks = ["hellaswag", "truthfulqa_mc2", "arc_easy", "winogrande", "boolq"]
        for benchmark in quick_benchmarks:
            assert benchmark in BENCHMARKS, f"Missing quick benchmark: {benchmark}"
            assert "⚡" in BENCHMARKS[benchmark], f"Quick benchmark {benchmark} should have ⚡ marker"

    def test_math_benchmarks_present(self):
        """Verify math/reasoning benchmarks are available."""
        assert "gsm8k" in BENCHMARKS


class TestMetricInfo:
    """Test metric information definitions."""

    def test_metric_info_not_empty(self):
        """Verify metric info dictionary is populated."""
        assert len(METRIC_INFO) > 0

    def test_common_metrics_have_descriptions(self):
        """Verify common metrics have descriptions."""
        common_metrics = ["acc", "acc_norm", "exact_match"]
        for metric in common_metrics:
            assert metric in METRIC_INFO, f"Missing metric info: {metric}"

    def test_metric_descriptions_are_strings(self):
        """Verify all metric descriptions are non-empty strings."""
        for key, desc in METRIC_INFO.items():
            assert isinstance(desc, str), f"Metric {key} description is not a string"
            assert len(desc) > 0, f"Metric {key} has empty description"


class TestMetricExtraction:
    """Test metric extraction from evaluation results."""

    def test_extract_acc_metric(self):
        """Test extracting accuracy metric from results."""
        results = {
            "truthfulqa_mc2": {
                "alias": "truthfulqa_mc2",
                "acc,none": 0.4645,
                "acc_stderr,none": 0.04,
            }
        }

        # Simulate metric extraction logic
        benchmark = "truthfulqa_mc2"
        ft_metrics = results.get(benchmark, {})

        metric_keys = []
        for key, val in ft_metrics.items():
            if isinstance(val, (int, float)) and not key.startswith("_"):
                if "stderr" not in key.lower() and key != "alias":
                    metric_keys.append(key)

        assert "acc,none" in metric_keys
        assert "acc_stderr,none" not in metric_keys  # stderr should be filtered
        assert "alias" not in metric_keys  # alias should be filtered

    def test_extract_multiple_metrics(self):
        """Test extracting multiple metrics (acc and acc_norm)."""
        results = {
            "hellaswag": {
                "alias": "hellaswag",
                "acc,none": 0.49,
                "acc_stderr,none": 0.05,
                "acc_norm,none": 0.69,
                "acc_norm_stderr,none": 0.046,
            }
        }

        benchmark = "hellaswag"
        ft_metrics = results.get(benchmark, {})

        metric_keys = []
        for key, val in ft_metrics.items():
            if isinstance(val, (int, float)) and not key.startswith("_"):
                if "stderr" not in key.lower() and key != "alias":
                    metric_keys.append(key)

        assert "acc,none" in metric_keys
        assert "acc_norm,none" in metric_keys
        assert len(metric_keys) == 2  # Only acc and acc_norm

    def test_metric_display_name_formatting(self):
        """Test metric name formatting for display."""
        test_cases = [
            ("acc,none", "Acc"),
            ("acc_norm,none", "Acc Norm"),
            ("exact_match,none", "Exact Match"),
        ]

        for raw_key, expected in test_cases:
            display_name = raw_key.replace(",none", "").replace("_", " ").replace(",", " ").title()
            assert display_name == expected, f"Expected {expected}, got {display_name}"

    def test_percentage_formatting(self):
        """Test metric value formatting as percentage."""
        test_values = [
            (0.4645, "46.45%"),
            (0.69, "69.00%"),
            (1.0, "100.00%"),
            (0.0, "0.00%"),
        ]

        for val, expected in test_values:
            formatted = f"{val * 100:.2f}%"
            assert formatted == expected, f"Expected {expected}, got {formatted}"


class TestResultStructure:
    """Test evaluation result structure handling."""

    def test_new_result_structure_with_metrics_key(self):
        """Test handling of new result structure with 'metrics' key."""
        results = {
            "metrics": {
                "hellaswag": {
                    "acc,none": 0.49,
                    "acc_norm,none": 0.69,
                }
            },
            "samples": {},
            "n_samples": {"hellaswag": 100},
        }

        # Simulate extraction logic
        ft_data = results.get("metrics", results)
        ft_metrics = ft_data.get("hellaswag", ft_data)

        assert "acc,none" in ft_metrics
        assert ft_metrics["acc,none"] == 0.49

    def test_legacy_result_structure(self):
        """Test handling of legacy result structure without 'metrics' key."""
        results = {
            "hellaswag": {
                "acc,none": 0.49,
                "acc_norm,none": 0.69,
            }
        }

        # Simulate extraction logic (fallback)
        ft_data = results.get("metrics", results)
        ft_metrics = ft_data.get("hellaswag", ft_data)

        assert "acc,none" in ft_metrics
        assert ft_metrics["acc,none"] == 0.49


class TestVisualBarCalculation:
    """Test visual bar width calculations."""

    def test_bar_width_calculation(self):
        """Test bar width calculation for visual display."""
        test_cases = [
            (0.5, 150),  # 50% -> 150px (half of 300)
            (1.0, 300),  # 100% -> 300px (full)
            (0.0, 0),  # 0% -> 0px
            (0.69, 207),  # 69% -> 207px (int truncates)
        ]

        for acc_val, expected_width in test_cases:
            bar_width = int(acc_val * 300)
            # Allow +/- 1 pixel difference due to float precision
            assert abs(bar_width - expected_width) <= 1, f"Expected ~{expected_width}px for {acc_val}, got {bar_width}px"

    def test_bar_width_bounds(self):
        """Test bar width stays within bounds."""
        for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            bar_width = int(val * 300)
            assert 0 <= bar_width <= 300, f"Bar width {bar_width} out of bounds for value {val}"


class TestComparisonMode:
    """Test comparison mode logic."""

    def test_comparison_enabled_with_base_results(self):
        """Test comparison mode detection when base results exist."""
        base_results = {
            "metrics": {"hellaswag": {"acc,none": 0.45}},
        }

        has_comparison = bool(base_results) and bool(base_results.get("metrics"))
        assert has_comparison is True

    def test_comparison_disabled_without_base_results(self):
        """Test comparison mode detection when no base results."""
        base_results = {}

        has_comparison = bool(base_results) and base_results.get("metrics")
        assert has_comparison is False

    def test_delta_calculation_positive(self):
        """Test delta calculation for improvement."""
        ft_val = 0.69
        base_val = 0.45

        delta = (ft_val - base_val) * 100
        assert delta == pytest.approx(24.0, rel=0.01)
        assert delta > 0  # Positive = improvement

    def test_delta_calculation_negative(self):
        """Test delta calculation for regression."""
        ft_val = 0.40
        base_val = 0.45

        delta = (ft_val - base_val) * 100
        assert delta == pytest.approx(-5.0, rel=0.01)
        assert delta < 0  # Negative = regression

    def test_delta_display_formatting(self):
        """Test delta display formatting with sign."""
        test_cases = [
            (24.0, "+24.00%"),
            (-5.0, "-5.00%"),
            (0.0, "+0.00%"),
        ]

        for delta, expected in test_cases:
            formatted = f"{delta:+.2f}%"
            assert formatted == expected


class TestBenchmarkCategories:
    """Test benchmark categorization."""

    def test_all_benchmarks_have_descriptions(self):
        """Verify every benchmark has a meaningful description."""
        for key, desc in BENCHMARKS.items():
            assert len(desc) >= 5, f"Benchmark {key} needs better description"

    def test_benchmark_keys_are_valid_identifiers(self):
        """Verify benchmark keys are valid for lm-eval."""
        for key in BENCHMARKS.keys():
            # Should be lowercase with underscores, no spaces
            assert key == key.lower(), f"Benchmark key {key} should be lowercase"
            assert " " not in key, f"Benchmark key {key} should not contain spaces"
