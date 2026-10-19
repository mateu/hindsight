"""Token-budget selection for the facts recall returns.

Recall ranks its candidates and then hands back as many as ``max_tokens`` worth
of fact text. Which facts make it into that answer is what this module decides —
the same job :mod:`hindsight_api.engine.source_facts` does for the provenance
map, and it answers to the same two properties (issues #3221, #3688):

* the budget is spent in **rank order**, so truncation hits the tail of the
  result list and the best-ranked facts are the ones that survive;
* one oversized fact skips itself only — it does not evict every shorter fact
  behind it.

Recall also keeps a floor: as long as the caller asked for a positive budget, a
run that matched something never answers with nothing. When no fact fits at all,
the top-ranked one is returned whole and over budget, because the alternatives
are worse — an empty answer reads as "this bank knows nothing about that", and a
fact clipped mid-sentence would be a claim the memory never made.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class FactSelection:
    """The facts that fit the budget, plus what the budget cost."""

    ids: list[str]
    """Selected fact IDs, in rank order (observations may inherit a source rank)."""

    total_tokens: int
    """Tokens the selected facts spend. Exceeds the budget when the floor applied."""

    truncated: bool
    """True when at least one ranked fact was dropped to stay inside the budget.

    Diagnostics only — it feeds the recall log line and trace. It is not worth
    reporting to callers: a broad candidate set usually carries more text than any
    sane budget, so on a bank of any size this is true on nearly every recall.
    """


def select_facts_within_budget(
    *,
    fact_ids_ordered: list[str],
    text_by_id: dict[str, str],
    max_tokens: int,
    count_tokens: Callable[[str], int],
    observation_sources: dict[str, list[str]] | None = None,
    candidate_limit: int | None = None,
) -> FactSelection:
    """Pick the ranked facts that fit ``max_tokens``.

    ``fact_ids_ordered`` must be in rank order — the budget is spent front to
    back, so their order decides which facts survive when it runs out. IDs absent
    from ``text_by_id`` are skipped without counting as truncation: those rows did
    not resolve at all, which is a different condition from running out of budget.

    A non-positive ``max_tokens`` selects nothing and the floor does not apply:
    ``max_tokens=0`` is the documented way to ask for chunks without facts
    (issue #364).

    ``observation_sources`` enables provenance-based observation preference. Only
    observations supplied in this mapping can replace sources. A fitting
    observation inherits its earliest source's selection position; if it cannot
    fit, the source remains eligible. Merely citing a source never suppresses it.
    This is a provenance policy, not a claim of semantic equivalence.

    ``candidate_limit`` bounds the ranked candidate window before token filtering.
    Selected observations free their sources' positions for backfill from the tail;
    facts skipped for exceeding the token budget do not. Only observations supplied
    in ``observation_sources`` can replace sources, so callers can keep replacement
    eligibility bounded to their original window while allowing source backfill.
    """
    if candidate_limit is not None and candidate_limit <= 0:
        return FactSelection(ids=[], total_tokens=0, truncated=False)

    if observation_sources:
        return _select_with_observation_preference(
            fact_ids_ordered=fact_ids_ordered,
            text_by_id=text_by_id,
            max_tokens=max_tokens,
            count_tokens=count_tokens,
            observation_sources=observation_sources,
            candidate_limit=candidate_limit,
        )

    if candidate_limit is not None:
        fact_ids_ordered = fact_ids_ordered[:candidate_limit]

    selected: list[str] = []
    total_tokens = 0
    skipped = False

    for fact_id in fact_ids_ordered:
        text = text_by_id.get(fact_id)
        if text is None:
            continue
        fact_tokens = count_tokens(text)
        if total_tokens + fact_tokens > max_tokens:
            skipped = True
            continue
        total_tokens += fact_tokens
        selected.append(fact_id)

    if not selected and skipped and max_tokens > 0:
        # Every candidate is individually over budget. Answering with nothing would be
        # indistinguishable from "no such memory", so spend the floor on the top-ranked
        # fact — the returned `total_tokens` shows the overshoot.
        for fact_id in fact_ids_ordered:
            text = text_by_id.get(fact_id)
            if text is not None:
                selected.append(fact_id)
                total_tokens = count_tokens(text)
                break

    return FactSelection(ids=selected, total_tokens=total_tokens, truncated=skipped)


def _select_with_observation_preference(
    *,
    fact_ids_ordered: list[str],
    text_by_id: dict[str, str],
    max_tokens: int,
    count_tokens: Callable[[str], int],
    observation_sources: dict[str, list[str]],
    candidate_limit: int | None,
) -> FactSelection:
    """Substitute and pack atomically, retaining raw evidence on budget failure."""
    costs = {fid: count_tokens(text_by_id[fid]) for fid in fact_ids_ordered if fid in text_by_id}
    if max_tokens <= 0:
        return FactSelection(
            ids=[],
            total_tokens=0,
            truncated=any(fid in costs for fid in fact_ids_ordered[:candidate_limit]),
        )

    sources = {
        oid: set(observation_sources[oid]) for oid in fact_ids_ordered if oid in observation_sources and oid in costs
    }
    replacements: dict[str, list[str]] = {}
    for oid, source_ids in sources.items():
        for sid in source_ids:
            replacements.setdefault(sid, []).append(oid)

    selected: list[str] = []
    superseded: set[str] = set()
    total_tokens = 0
    skipped = False
    candidates_considered = 0
    first_candidate: str | None = None

    selected_ids: set[str] = set()

    def try_select(fid: str) -> bool:
        nonlocal total_tokens
        if fid in selected_ids:
            return True
        if total_tokens + costs[fid] > max_tokens:
            return False
        selected.append(fid)
        selected_ids.add(fid)
        total_tokens += costs[fid]
        superseded.update(sources.get(fid, set()))
        return True

    # Try each covering observation at its earliest source's position. If it
    # does not fit there, it cannot fit later as the budget only shrinks. Thus a
    # selected observation never needs to evict an already selected raw source,
    # and duplicate text never consumes tokens that need a second packing pass.
    for fid in fact_ids_ordered:
        if fid in selected_ids or fid in superseded:
            continue
        if candidate_limit is not None and candidates_considered >= candidate_limit:
            break
        # Budget failures still occupy a candidate position. Only successful
        # substitutions (including the promoted observation's later duplicate)
        # make room for lower-ranked candidates beyond the original window.
        candidates_considered += 1
        if fid not in costs:
            continue
        if first_candidate is None:
            first_candidate = fid
        if any(try_select(oid) for oid in replacements.get(fid, [])):
            continue
        if not try_select(fid):
            skipped = True

    if not selected and first_candidate is not None:
        # Preserve the existing positive-budget floor, including its explicit
        # overshoot accounting. Do not force an oversized replacement instead.
        selected = [first_candidate]
        total_tokens = costs[first_candidate]

    return FactSelection(ids=selected, total_tokens=total_tokens, truncated=skipped)
