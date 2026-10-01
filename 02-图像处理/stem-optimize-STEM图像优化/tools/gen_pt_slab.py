"""Generate a Pt-terminated Cu/Pt VASP slab without import-time side effects."""

import argparse
import shutil
from pathlib import Path

import numpy as np

SUPPORT_FILES = ("INCAR", "KPOINTS", "POTCAR", "vasp.slurm")


def generate_slab(source: Path, destination: Path, overwrite: bool = False):
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"源目录不存在: {source}")
    if source == destination:
        raise ValueError("源目录与目标目录不能相同")

    targets = [destination / "POSCAR"]
    targets.extend(
        destination / name for name in SUPPORT_FILES if (source / name).exists()
    )
    conflicts = [path for path in targets if path.exists()]
    if conflicts and not overwrite:
        raise FileExistsError(
            "目标文件已存在；使用 --overwrite 才能替换:\n"
            + "\n".join(str(path) for path in conflicts)
        )
    destination.mkdir(parents=True, exist_ok=True)

    for name in SUPPORT_FILES:
        source_file = source / name
        if source_file.exists():
            shutil.copy2(source_file, destination / name)

    a_fcc = 2.7322010057402428 * np.sqrt(2)
    d111 = a_fcc / np.sqrt(3)
    in_plane = a_fcc / np.sqrt(2) * 2
    a1 = np.array([in_plane, 0.0, 0.0])
    a2 = np.array([in_plane * 0.5, in_plane * np.sqrt(3) / 2, 0.0])
    layer_count = 6
    vacuum = 15.0
    a3 = np.array([0.0, 0.0, (layer_count - 1) * d111 + vacuum])

    stack_offsets = {
        "A": np.array([0.0, 0.0]),
        "B": np.array([1.0 / 3, 2.0 / 3]),
        "C": np.array([2.0 / 3, 1.0 / 3]),
    }
    supercell_offsets = [(i / 2.0, j / 2.0) for i in range(2) for j in range(2)]
    in_plane_matrix = np.column_stack([a1[:2], a2[:2]])
    stack_sequence = ["A", "B", "C", "A", "B", "C"]
    layer_elements = ["Cu", "Pt", "Cu", "Pt", "Cu", "Pt"]

    atoms = []
    for layer in range(layer_count):
        offset = stack_offsets[stack_sequence[layer]]
        element = layer_elements[layer]
        for du, dv in supercell_offsets:
            uv = np.array([(offset[0] + du) % 1.0, (offset[1] + dv) % 1.0])
            xy = in_plane_matrix @ uv
            atoms.append((np.array([xy[0], xy[1], layer * d111]), element, layer))

    copper = [(coords, layer) for coords, elem, layer in atoms if elem == "Cu"]
    platinum = [(coords, layer) for coords, elem, layer in atoms if elem == "Pt"]
    fixed_layers = {0, 1}
    with (destination / "POSCAR").open("w", encoding="ascii", newline="\n") as stream:
        stream.write("L11 CuPt(111) Pt-terminated 6L p(2x2)\n1.0\n")
        for vector in (a1, a2, a3):
            stream.write(
                f"  {vector[0]:16.10f}  {vector[1]:16.10f}  {vector[2]:16.10f}\n"
            )
        stream.write(f"Cu Pt\n  {len(copper)}  {len(platinum)}\n")
        stream.write("Selective dynamics\nCartesian\n")
        for coords, layer in copper + platinum:
            flags = "F F F" if layer in fixed_layers else "T T T"
            stream.write(
                f"  {coords[0]:16.10f}  {coords[1]:16.10f}  "
                f"{coords[2]:16.10f}  {flags}\n"
            )
    return {
        "destination": destination,
        "copper_atoms": len(copper),
        "platinum_atoms": len(platinum),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许替换目标目录中的同名文件",
    )
    args = parser.parse_args()
    result = generate_slab(args.source, args.destination, overwrite=args.overwrite)
    print(
        f"完成: {result['destination']} | "
        f"Cu={result['copper_atoms']} Pt={result['platinum_atoms']}"
    )


if __name__ == "__main__":
    main()
