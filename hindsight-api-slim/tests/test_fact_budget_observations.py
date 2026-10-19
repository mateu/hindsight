"""Observation preference must preserve deliverable evidence through packing."""

import pytest

from hindsight_api.engine.fact_budget import select_facts_within_budget


def choose(texts: dict[str, int], sources: dict[str, list[str]], budget: int = 1024):
    return select_facts_within_budget(
        fact_ids_ordered=list(texts),
        text_by_id={fid: "x " * size for fid, size in texts.items()},
        max_tokens=budget,
        count_tokens=lambda text: len(text.split()),
        observation_sources=sources,
    )


def test_rank_one_answer_survives_when_rank_21_observation_cannot_fit():
    texts = {"answer": 20, **{f"noise-{i}": 60 for i in range(19)}, "observation": 1100}
    result = choose(texts, {"observation": ["answer"]})
    assert result.ids[0] == "answer"
    assert "observation" not in result.ids
    assert result.total_tokens <= 1024


def test_fitting_observation_inherits_source_position_before_noise_spends_budget():
    result = choose({"answer": 20, "noise": 1000, "observation": 100}, {"observation": ["answer"]})
    assert result.ids == ["observation"]
    assert result.total_tokens == 100


def test_source_fallback_when_preceding_facts_leave_too_little_room():
    result = choose({"first": 1000, "answer": 20, "observation": 40}, {"observation": ["answer"]})
    assert result.ids == ["first", "answer"]
    assert result.total_tokens == 1020


def test_observation_replaces_multiple_sources_without_duplicate_token_cost():
    result = choose({"a": 20, "b": 20, "other": 984, "observation": 40}, {"observation": ["a", "b"]})
    assert result.ids == ["observation", "other"]
    assert result.total_tokens == 1024
    assert not result.truncated


def test_an_oversized_first_cover_does_not_block_a_fitting_second_cover():
    result = choose(
        {"answer": 3, "large-obs": 9, "small-obs": 4}, {"large-obs": ["answer"], "small-obs": ["answer"]}, 4
    )
    assert result.ids == ["small-obs"]
    assert result.total_tokens == 4


def test_overlapping_observations_are_deterministic_and_do_not_drop_unrelated_raw():
    result = choose({"a": 2, "unrelated": 2, "obs1": 3, "obs2": 3}, {"obs1": ["a"], "obs2": ["a"]}, 8)
    assert result.ids == ["obs1", "unrelated", "obs2"]
    assert result.total_tokens == 8


def test_missing_or_out_of_window_observation_cannot_suppress_source():
    result = choose({"answer": 20}, {"not-retrieved": ["answer"]})
    assert result.ids == ["answer"]


def test_missing_observation_text_cannot_suppress_source():
    result = select_facts_within_budget(
        fact_ids_ordered=["answer", "obs"],
        text_by_id={"answer": "answer"},
        max_tokens=1,
        count_tokens=lambda text: len(text.split()),
        observation_sources={"obs": ["answer"]},
    )
    assert result.ids == ["answer"]


@pytest.mark.parametrize("budget", [0, -1])
def test_nonpositive_budget_disables_even_zero_token_facts(budget):
    assert choose({"answer": 0, "obs": 0}, {"obs": ["answer"]}, budget).ids == []


def test_existing_floor_returns_highest_ranked_evidence_not_oversized_replacement():
    result = choose({"answer": 20, "obs": 40}, {"obs": ["answer"]}, 10)
    assert result.ids == ["answer"]
    assert result.total_tokens == 20
    assert result.truncated


def test_observation_only_control_keeps_rank_order():
    result = choose({"obs1": 3, "obs2": 2}, {"obs1": ["unretrieved-source"]}, 5)
    assert result.ids == ["obs1", "obs2"]


def test_no_preference_preserves_raw_and_observation():
    result = choose({"answer": 20, "obs": 40}, {}, 60)
    assert result.ids == ["answer", "obs"]
