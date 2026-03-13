"""
Tests for the Query Expansion Learning Loop.

Verifies:
  1. Mock dataset generation produces expected shapes
  2. CTR calculation is correct
  3. Learning loop adds new expansion terms
  4. Learning loop produces CTR improvement on training data (overfitting sanity check)
  5. Term weights stay within valid bounds
  6. Dead terms are pruned
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.learning_loop import (
    BRAND_DEFINITIONS,
    POSITIVE_ACTIONS,
    FeedbackEvent,
    QueryTerm,
    calculate_ctr,
    generate_mock_dataset,
    initialize_query_terms,
    run_query_expansion_loop,
    simulate_updated_ctr,
)


@pytest.fixture(scope="module")
def dataset():
    posts, alerts, feedbacks = generate_mock_dataset(seed=42)
    return posts, alerts, feedbacks


@pytest.fixture(scope="module")
def baseline_terms():
    return initialize_query_terms()


@pytest.fixture(scope="module")
def updated_terms(dataset, baseline_terms):
    posts, alerts, feedbacks = dataset
    return run_query_expansion_loop(
        posts=posts,
        alerts=alerts,
        feedbacks=feedbacks,
        current_terms=baseline_terms,
    )


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestMockDataset:
    def test_post_count(self, dataset):
        posts, _, _ = dataset
        # 3 brands × (5 positive × 3 variants + 2 negative × 2 variants) = 57 posts
        assert len(posts) == 57

    def test_alert_count_equals_post_count(self, dataset):
        posts, alerts, _ = dataset
        assert len(alerts) == len(posts)

    def test_feedback_count_equals_alert_count(self, dataset):
        _, alerts, feedbacks = dataset
        assert len(feedbacks) == len(alerts)

    def test_all_brands_represented(self, dataset):
        posts, _, _ = dataset
        brand_ids = {p.brand_id for p in posts}
        assert brand_ids == set(BRAND_DEFINITIONS.keys())

    def test_feedback_actions_are_valid(self, dataset):
        _, _, feedbacks = dataset
        valid_actions = {"click", "dismiss", "content_brief", "share"}
        for fb in feedbacks:
            assert fb.action in valid_actions

    def test_positive_feedback_exists(self, dataset):
        _, _, feedbacks = dataset
        positive = [fb for fb in feedbacks if fb.action in POSITIVE_ACTIONS]
        assert len(positive) > 0


# ---------------------------------------------------------------------------
# CTR calculation tests
# ---------------------------------------------------------------------------

class TestCalculateCTR:
    def test_ctr_in_valid_range(self, dataset):
        posts, alerts, feedbacks = dataset
        ctr = calculate_ctr(alerts, feedbacks)
        assert 0.0 <= ctr <= 1.0

    def test_ctr_per_brand_in_valid_range(self, dataset):
        _, alerts, feedbacks = dataset
        for brand_id in BRAND_DEFINITIONS:
            ctr = calculate_ctr(alerts, feedbacks, brand_id=brand_id)
            assert 0.0 <= ctr <= 1.0, f"CTR out of range for {brand_id}: {ctr}"

    def test_ctr_zero_when_no_feedback(self, dataset):
        _, alerts, _ = dataset
        ctr = calculate_ctr(alerts, [])  # empty feedbacks
        assert ctr == 0.0

    def test_ctr_manual_calculation(self):
        """Verify CTR formula on a controlled mini-dataset."""
        from src.learning_loop import Alert
        from datetime import datetime

        alerts = [
            Alert("a1", "brand_x", "p1", 0.8, datetime.now()),
            Alert("a2", "brand_x", "p2", 0.7, datetime.now()),
            Alert("a3", "brand_x", "p3", 0.6, datetime.now()),
            Alert("a4", "brand_x", "p4", 0.5, datetime.now()),
        ]
        feedbacks = [
            FeedbackEvent("f1", "a1", "brand_x", "p1", "click", 5000, datetime.now()),
            FeedbackEvent("f2", "a2", "brand_x", "p2", "dismiss", 500, datetime.now()),
            FeedbackEvent("f3", "a3", "brand_x", "p3", "content_brief", 30000, datetime.now()),
            FeedbackEvent("f4", "a4", "brand_x", "p4", "dismiss", 400, datetime.now()),
        ]
        # 2 positive (click + content_brief) out of 4 = 0.5
        ctr = calculate_ctr(alerts, feedbacks)
        assert abs(ctr - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Learning loop tests
# ---------------------------------------------------------------------------

class TestRunQueryExpansionLoop:
    def test_returns_all_brands(self, updated_terms):
        assert set(updated_terms.keys()) == set(BRAND_DEFINITIONS.keys())

    def test_core_terms_preserved(self, baseline_terms, updated_terms):
        for brand_id in BRAND_DEFINITIONS:
            baseline_core = {
                qt.term for qt in baseline_terms[brand_id] if qt.term_type == "core"
            }
            updated_core = {
                qt.term for qt in updated_terms[brand_id] if qt.term_type == "core"
            }
            assert baseline_core == updated_core, (
                f"Core terms changed for {brand_id}: "
                f"removed={baseline_core - updated_core}, added={updated_core - baseline_core}"
            )

    def test_new_expansion_terms_added(self, baseline_terms, updated_terms):
        total_new = 0
        for brand_id in BRAND_DEFINITIONS:
            baseline_count = len(baseline_terms[brand_id])
            updated_count = len(updated_terms[brand_id])
            total_new += updated_count - baseline_count
        assert total_new > 0, "Learning loop added no new expansion terms"

    def test_expansion_term_weights_in_valid_range(self, updated_terms):
        for brand_id, terms in updated_terms.items():
            for qt in terms:
                assert 0.0 <= qt.weight <= 1.0, (
                    f"Term '{qt.term}' for {brand_id} has invalid weight {qt.weight}"
                )

    def test_expansion_term_source_is_learning_loop(self, updated_terms):
        for brand_id, terms in updated_terms.items():
            for qt in terms:
                if qt.term_type == "expansion":
                    assert qt.source == "learning_loop"

    def test_no_terms_below_prune_threshold(self, updated_terms):
        """Terms with weight < 0.25 should have been pruned."""
        for brand_id, terms in updated_terms.items():
            for qt in terms:
                assert qt.weight >= 0.25, (
                    f"Term '{qt.term}' for {brand_id} below prune threshold: {qt.weight}"
                )


# ---------------------------------------------------------------------------
# Overfitting sanity check — the key test
# ---------------------------------------------------------------------------

class TestOverfitting:
    def test_ctr_improves_on_training_data(self, dataset, updated_terms):
        """
        Core sanity check: the learning loop must show measurable CTR improvement
        on the training data it was optimized on.
        This demonstrates the learning mechanism actually works.
        """
        posts, alerts, feedbacks = dataset

        ctr_before = calculate_ctr(alerts, feedbacks)
        ctr_after, fraction_surfaced = simulate_updated_ctr(
            posts=posts,
            alerts=alerts,
            feedbacks=feedbacks,
            updated_terms=updated_terms,
            score_threshold=0.55,
        )

        assert ctr_after > ctr_before, (
            f"CTR did not improve: before={ctr_before:.3f}, after={ctr_after:.3f}"
        )

    def test_ctr_improvement_is_meaningful(self, dataset, updated_terms):
        """CTR improvement should be at least 5 percentage points on training data."""
        posts, alerts, feedbacks = dataset

        ctr_before = calculate_ctr(alerts, feedbacks)
        ctr_after, _ = simulate_updated_ctr(
            posts=posts,
            alerts=alerts,
            feedbacks=feedbacks,
            updated_terms=updated_terms,
            score_threshold=0.55,
        )

        improvement = ctr_after - ctr_before
        assert improvement >= 0.05, (
            f"CTR improvement too small: {improvement:.3f} (expected >= 0.05)"
        )

    def test_fraction_surfaced_is_reasonable(self, dataset, updated_terms):
        """Updated terms should surface a reasonable fraction of alerts (not 0 or 100%)."""
        posts, alerts, feedbacks = dataset

        _, fraction_surfaced = simulate_updated_ctr(
            posts=posts,
            alerts=alerts,
            feedbacks=feedbacks,
            updated_terms=updated_terms,
            score_threshold=0.55,
        )

        assert 0.05 < fraction_surfaced < 1.0, (
            f"Unexpected fraction surfaced: {fraction_surfaced:.3f}"
        )
