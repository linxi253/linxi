# Legacy material

This directory is not imported by the v5 Python package and is excluded from
the v5 PyInstaller specifications.

- `imagej/` contains the historical FFT macro that builds an FFT movie from a
  stack.
- `pyinstaller-specs/` contains obsolete specifications whose entry scripts no
  longer exist.

The full pre-repair directory was additionally archived on the original
developer machine as `<仓库根>\08-历史版本\hrtem-HRTEM滤波工具-v4-2026-07-29`;
that archive is **not distributed with this repository**.

## Third-party files not redistributed here

The following files existed in the local working copy but are **not included in
this repository**, because their upstream authors have not granted
redistribution terms:

| File | Author | Version |
|---|---|---|
| `digitalmicrograph/HRTEM Filter.s` | D. R. G. Mitchell | v2.0 (2014) |
| `digitalmicrograph/去噪音插件.s` / `.txt` | D. R. G. Mitchell | v4.0 (2019), <https://www.dmscripting.com> |
| `imagej/Radial_Profile_Angle.jar` | Philippe Carl | v1.2 (2019) |

They remain available from their original sources. The maintained v5
implementation of the same algorithms lives in `butter.py` and the
`hrtem_filter` package.
