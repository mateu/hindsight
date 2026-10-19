"""Observation preference must preserve deliverable evidence through packing."""

import pytest

from hindsight_api.engine.fact_budget import select_facts_within_budget


def choose(
    texts: dict[str, int],
    sources: dict[str, list[str]],
    budget: int = 1024,
    candidate_limit: int | None = None,
):
    return select_facts_within_budget(
        fact_ids_ordered=list(texts),
        text_by_id={fid: "x " * size for fid, size in texts.items()},
        max_tokens=budget,
        count_tokens=lambda text: len(text.split()),
        observation_sources=sources,
        candidate_limit=candidate_limit,
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


def test_selected_observation_backfills_201st_candidate_into_200_item_window():
    texts = {"answer": 1, "obs": 1, **{f"other-{i}": 1 for i in range(198)}, "tail": 1}

    result = choose(texts, {"obs": ["answer"]}, candidate_limit=200)

    assert result.ids == ["obs", *[f"other-{i}" for i in range(198)], "tail"]
    assert result.total_tokens == 200
    assert not result.truncated


def test_oversized_candidates_and_failed_replacement_do_not_free_window_positions():
    result = choose(
        {"large": 9, "answer": 1, "obs": 4, "tail": 1},
        {"obs": ["answer"]},
        budget=2,
        candidate_limit=3,
    )

    assert result.ids == ["answer"]
    assert result.total_tokens == 1
    assert result.truncated


def test_backfilled_observation_without_replacement_eligibility_keeps_its_rank():
    # Only obs-in was in the original three-candidate window. A later observation
    # may backfill, but its provenance must not promote it or suppress raw evidence.
    result = choose(
        {"answer": 1, "obs-in": 2, "other": 1, "obs-out": 2, "tail": 1},
        {"obs-in": ["answer"]},
        candidate_limit=3,
    )

    assert result.ids == ["obs-in", "other", "obs-out"]
    assert result.total_tokens == 5


def test_partial_overlap_suppresses_backfill_sources_beyond_the_original_window():
    result = choose(
        {"a": 1, "b": 1, "obs1": 2, "obs2": 2, "later-source": 1, "tail1": 1, "tail2": 1},
        {"obs1": ["a"], "obs2": ["a", "b", "later-source"]},
        candidate_limit=4,
    )

    assert result.ids == ["obs1", "obs2", "tail1", "tail2"]
    assert result.total_tokens == 6
    assert not result.truncated


def test_unselected_cover_does_not_suppress_a_source_beyond_the_original_window():
    result = choose(
        {"answer": 1, "obs1": 1, "obs2": 9, "later-source": 1, "tail": 1},
        {"obs1": ["answer"], "obs2": ["answer", "later-source"]},
        budget=3,
        candidate_limit=3,
    )

    assert result.ids == ["obs1", "later-source"]
    assert result.total_tokens == 2
    assert result.truncated


def test_preference_off_still_respects_the_original_candidate_window():
    result = choose({"large": 9, "answer": 1, "tail": 1}, {}, budget=2, candidate_limit=2)

    assert result.ids == ["answer"]
    assert result.total_tokens == 1
    assert result.truncated


@pytest.mark.parametrize("sources", [{}, {"obs": ["answer"]}])
def test_missing_text_does_not_free_a_candidate_position(sources):
    result = select_facts_within_budget(
        fact_ids_ordered=["missing", "answer", "obs"],
        text_by_id={"answer": "answer", "obs": "observation"},
        max_tokens=1,
        count_tokens=lambda text: len(text.split()),
        observation_sources=sources,
        candidate_limit=1,
    )

    assert result.ids == []
    assert result.total_tokens == 0
    assert not result.truncated


@pytest.mark.parametrize("sources", [{}, {"obs": ["answer"]}])
def test_empty_candidate_window_disables_the_floor(sources):
    result = choose({"answer": 20, "obs": 40}, sources, budget=10, candidate_limit=0)

    assert result.ids == []
    assert result.total_tokens == 0
    assert not result.truncated


@pytest.mark.parametrize("budget", [0, -1])
def test_nonpositive_budget_keeps_backfill_disabled(budget):
    result = choose({"answer": 1, "obs": 1, "tail": 1}, {"obs": ["answer"]}, budget, candidate_limit=2)

    assert result.ids == []
    assert result.total_tokens == 0
    assert result.truncated
