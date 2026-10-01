from pathlib import Path

import pytest

from atom_center.splitting import (
    grouped_split_records,
    read_manifest_csv,
    split_manifest_csv,
    write_manifest_csv,
)


def sample_records(group_count: int = 10) -> list[dict[str, str]]:
    return [
        {
            "image_id": f"image-{group}-{index}",
            "acquisition_id": f"acq-{group}",
            "modality": "haadf_stem",
        }
        for group in range(group_count)
        for index in range(2)
    ]


def test_group_split_is_deterministic_and_has_no_leakage() -> None:
    records = sample_records()
    first = grouped_split_records(records, seed=123)
    second = grouped_split_records(records, seed=123)
    assert dict(first.group_to_split) == dict(second.group_to_split)
    assert set(first.records_by_split) == {"train", "val", "test"}
    assert all(value > 0 for value in first.group_counts.values())
    for split_name, rows in first.records_by_split.items():
        assert all(first.group_to_split[row["acquisition_id"]] == split_name for row in rows)


def test_group_split_rejects_too_few_groups() -> None:
    with pytest.raises(ValueError):
        grouped_split_records(sample_records(group_count=2))


def test_group_split_rejects_missing_group_value() -> None:
    records = sample_records()
    records[0]["acquisition_id"] = None  # type: ignore[assignment]
    with pytest.raises(ValueError):
        grouped_split_records(records)


def test_manifest_csv_round_trip_and_split(tmp_path: Path) -> None:
    source = tmp_path / "manifest.csv"
    output = tmp_path / "split.csv"
    write_manifest_csv(source, sample_records())
    assert len(read_manifest_csv(source)) == 20
    split_manifest_csv(source, output, seed=42)
    rows = read_manifest_csv(output)
    assert len(rows) == 20
    assert {row["split"] for row in rows} == {"train", "val", "test"}
