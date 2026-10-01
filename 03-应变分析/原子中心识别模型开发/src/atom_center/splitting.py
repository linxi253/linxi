"""Leakage-resistant, deterministic dataset splitting by acquisition group."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

DEFAULT_RATIOS: Mapping[str, float] = MappingProxyType(
    {"train": 0.8, "val": 0.1, "test": 0.1}
)


@dataclass(frozen=True)
class GroupSplitResult:
    records_by_split: Mapping[str, tuple[dict[str, str], ...]]
    group_to_split: Mapping[str, str]

    @property
    def record_counts(self) -> dict[str, int]:
        return {name: len(records) for name, records in self.records_by_split.items()}

    @property
    def group_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.records_by_split}
        for split_name in self.group_to_split.values():
            counts[split_name] += 1
        return counts


def _validated_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    if not ratios:
        raise ValueError("at least one split ratio is required")
    normalized = {str(name): float(value) for name, value in ratios.items()}
    if any(value < 0.0 for value in normalized.values()):
        raise ValueError("split ratios cannot be negative")
    total = sum(normalized.values())
    if not np_isclose(total, 1.0):
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    return {name: value for name, value in normalized.items() if value > 0.0}


def np_isclose(left: float, right: float, tolerance: float = 1e-9) -> bool:
    """Tiny dependency-free scalar closeness helper."""

    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def grouped_split_records(
    records: Sequence[Mapping[str, object]],
    *,
    group_key: str = "acquisition_id",
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    seed: int = 20260831,
    min_groups_per_split: Mapping[str, int] | None = None,
) -> GroupSplitResult:
    """Assign whole groups to splits while approximately balancing record counts."""

    split_ratios = _validated_ratios(ratios)
    if not records:
        raise ValueError("records cannot be empty")
    copied_records = [
        {
            str(key): ("" if value is None else str(value))
            for key, value in row.items()
        }
        for row in records
    ]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, record in enumerate(copied_records):
        group = record.get(group_key, "").strip()
        if not group:
            raise ValueError(f"record {index} has no non-empty {group_key!r}")
        record[group_key] = group
        groups[group].append(record)
    minimums = {name: 1 for name in split_ratios}
    for name, count in (min_groups_per_split or {}).items():
        if name not in minimums or not isinstance(count, int) or count < 1:
            raise ValueError("minimum groups must name an active split and be positive integers")
        minimums[name] = count
    if len(groups) < sum(minimums.values()):
        raise ValueError(
            f"{len(groups)} groups cannot populate required split group counts {minimums}"
        )

    rng = random.Random(seed)
    group_items = sorted(groups.items())
    rng.shuffle(group_items)
    # Largest acquisition groups are placed first; the preceding shuffle gives
    # a deterministic random tie break for equally sized groups.
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    split_names = list(split_ratios)
    total_records = len(copied_records)
    targets = {
        name: split_ratios[name] * total_records for name in split_names
    }
    counts = {name: 0 for name in split_names}
    groups_per_split = {name: 0 for name in split_names}
    group_to_split: dict[str, str] = {}

    for item_index, (group_name, group_records) in enumerate(group_items):
        empty_splits = [name for name in split_names if groups_per_split[name] < minimums[name]]
        required_groups = sum(max(0, minimums[name]-groups_per_split[name]) for name in split_names)
        remaining_groups = len(group_items) - item_index
        candidates = (
            empty_splits if remaining_groups == required_groups else split_names
        )

        def deficit_priority(name: str) -> tuple[float, float, int]:
            target = max(targets[name], 1.0)
            relative_deficit = (targets[name] - counts[name]) / target
            # Stable final tie-break preserves the caller's split order.
            return (
                relative_deficit,
                split_ratios[name],
                -split_names.index(name),
            )

        selected = max(candidates, key=deficit_priority)
        group_to_split[group_name] = selected
        counts[selected] += len(group_records)
        groups_per_split[selected] += 1

    output: dict[str, list[dict[str, str]]] = {name: [] for name in split_names}
    for record in copied_records:
        output[group_to_split[record[group_key]]].append(record)

    return GroupSplitResult(
        records_by_split=MappingProxyType(
            {name: tuple(rows) for name, rows in output.items()}
        ),
        group_to_split=MappingProxyType(group_to_split),
    )


def read_manifest_csv(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"manifest has no header: {manifest_path}")
        return [dict(row) for row in reader]


def write_manifest_csv(
    path: str | Path,
    records: Iterable[Mapping[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    output_path = Path(path)
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("cannot write an empty manifest")
    if fieldnames is None:
        ordered_fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered_fields:
                    ordered_fields.append(str(key))
        fieldnames = ordered_fields
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def split_manifest_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    group_key: str = "acquisition_id",
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    seed: int = 20260831,
) -> GroupSplitResult:
    records = read_manifest_csv(input_path)
    result = grouped_split_records(
        records, group_key=group_key, ratios=ratios, seed=seed
    )
    combined: list[dict[str, str]] = []
    for split_name, split_records in result.records_by_split.items():
        combined.extend({**record, "split": split_name} for record in split_records)
    write_manifest_csv(output_path, combined)
    return result
