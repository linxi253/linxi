# AIforTEM — In-Situ Electron Microscopy Data Processing Toolkit

English | [中文](README.md)

This repository collects the source code and documentation of TEM, HRTEM, STEM, 4D-STEM
image-processing, EELS spectroscopy and quantitative-analysis tools that I developed,
maintained and packaged during graduate school.
The tools were originally independent (each with its own git repository); they are gathered
here under their original directory layout so everything can be browsed and cloned in one place.

> **This repository contains source code and documentation only.** Executables (`.exe`),
> model weights (`.pt`/`.onnx`), raw microscopy data (`.tif`/`.dm3`/`.dm4`) and PyInstaller
> build artifacts are far too large to commit — see [.gitignore](.gitignore).
> To get a working program, build it yourself following each tool's README.

## Tools at a Glance

Versions are taken from each tool's code constants (`__version__` / `pyproject.toml`), last verified 2026-09-28.

| Category | Tool | Version | Source directory | Entry point |
|---|---|---|---|---|
| Video & data extraction | TEM video frame extraction | 4.4 | `01-视频与数据提取/视频切片工具` | `run_app.py` |
| Video & data extraction | TEM video pipeline (slicing + drift correction) | — | `10-DSH集成` | `tem_pipeline.py` |
| Image processing | TIFF drift correction | 7.3.0 | `02-图像处理/drift-correction-v7` | `drift_correction.py` |
| Image processing | HRTEM/STEM filtering | 5.1.0 | `02-图像处理/hrtem-HRTEM滤波工具` | `hrtem.py`, `cli_launcher.py` |
| Image processing | HRTEM/STEM image enhancement | 2.1.0 | `02-图像处理/stem-optimize-STEM图像优化` | `main.py` |
| Image processing | TIF image filters | 1.2.0 | `02-图像处理/图像加滤镜工具` | `main.py` |
| Image processing | Delocalization removal | 1.1.0 | `02-图像处理/离域效应去除工具` | `main.py` |
| Strain analysis | Strain++ GPA ⚠️ GPL | 1.4.1 | `03-应变分析/strainpp-GPA应变分析` | `run.py` |
| Strain analysis | PPA atomic displacement & strain analysis | 3.4.0 | `03-应变分析/原子级应力分析-PPA` | `ppa.py` |
| Strain analysis | Atom identification & intensity analysis | 1.3.0 | `03-应变分析/原子识别纯算法` | `atomic_app.py` |
| Strain analysis | Atom-center recognition model (training/inference) | 0.3.0 | `03-应变分析/原子中心识别模型开发` | `src/atom_center` |
| Statistics | TIF atomic contrast statistics | 2.5.0 | `04-统计分析/原子衬度统计` | `tif图像衬度分析工具.py` |
| Statistics | HAADF-STEM feature-region evolution | 2026.09.2 | `04-统计分析/特征区域演化分析` | `应力面积统计.py` |
| Statistics | TIFF area measurement | — | `04-统计分析/统计面积` | `统计面积.py` |
| Statistics | Crystalline/amorphous region statistics | 5.3 | `04-统计分析/非晶面积统计/pythonProject` | `main.py` |
| 4D-STEM | 4D-STEM Processor | 2.2.0 | `05-4D-STEM分析/4D-STEM-Processor` | `stem_processor_gui.py` |
| EELS spectroscopy | EELS edge valence analysis | 0.2.0 | `05-EELS分析/EELS边缘价态分析工具` | `run.py` |
| HRTEM simulation | HRTEM multislice simulation (tem_sim engine) | 1.2.0 | `09-HRTEM模拟` | `run.bat` |
| STEM simulation | STEM-HAADF simulation (stem_sim engine, frozen-phonon multislice) | 0.1.0 | `010-STEM模拟` | `run.bat` |
| Integration | TEM Suite (integrated GUI) | 1.1.0 | `全整合` | `run.py` |

`—` means the tool keeps no version constant in its code.

## Directory Guide

- [01-视频与数据提取](01-视频与数据提取/README.md): video → ImageJ TIFF; FFmpeg and XRD operation screen recordings (the latter two are local-only and not committed).
- [02-图像处理](02-图像处理/README.md): drift correction, HRTEM/STEM enhancement, delocalization removal and general TIF filters.
- [03-应变分析](03-应变分析/README.md): GPA and PPA strain analysis, atom identification, and the atom-center recognition model engineering.
- [04-统计分析](04-统计分析/README.md): contrast, region evolution, manual area measurement and crystalline/amorphous statistics.
- [05-4D-STEM分析](05-4D-STEM分析/README.md): DPC, SSB, orientation, strain and ptychography on DM4 data.
- [05-EELS分析](05-EELS分析/README.md): Dual-EELS edge valence quantification (MLLS fitting, plural-scattering correction, bootstrap).
- [06-独立脚本](06-独立脚本/README.md): denoising and setup scripts not packaged as projects.
- [09-HRTEM模拟](09-HRTEM模拟/README.md): multislice HRTEM simulation GUI with CIF import and controllable zone axis / microscope parameters / thickness / orientation.
- [010-STEM模拟](010-STEM模拟/README.md): HAADF/ADF/BF/ABF STEM simulation GUI with CIF import and controllable zone axis / thickness / probe / annular detectors.
- [10-DSH集成](10-DSH集成/README.md): TEM video pipeline driven inside DeepSeek Harness sessions.
- [全整合](全整合/README.md): source of the TEM Suite integration layer (tool tree + tabs).

> Numbering note: numbers reflect the order in which directories joined the workspace and are
> **not contiguous** — `07` and `08` were never created (`08-历史版本` in some older documents
> refers to an off-repository local archive; no such directory exists in the repository).
> There are two `05-` directories (4D-STEM and EELS) for historical reasons; they were not
> renumbered, to avoid breaking relative references inside each tool. `010-STEM模拟` uses a
> three-digit number meaning "group 10": it sorts lexicographically between `01-` and `02-`,
> and is unrelated to `10-DSH集成` (a STEM simulation engine vs. a TEM video pipeline); the
> name is kept for compatibility with hard-coded paths (全整合/temsuite/registry.py, .tools, CI).

## Running from Source

**Always run each tool with its own virtual environment**, never the `python` on PATH:
the tools pin conflicting dependency versions (numpy 1.26 and 2.2 coexist, as do Python 3.10 and 3.12).
A shared interpreter would let a tool import another project's versions, and `pip install`
would pollute the global environment and break the other tools.

First-time setup for each project (once only):

```powershell
cd <project directory>
python -m venv .venv
.venv\Scripts\python -X utf8 -m pip install -r requirements.lock.txt
```

Then always launch with the project's own interpreter:

```powershell
.venv\Scripts\python <entry script.py>
```

`09-HRTEM模拟` and `010-STEM模拟` provide a `run.bat`; `原子识别纯算法`'s launcher is `启动原子识别工具.bat` — double-click to launch.

Projects with a `.spec` file can be re-packaged with PyInstaller **inside the project's own environment**:

```powershell
.venv\Scripts\python -m pip install pyinstaller
.venv\Scripts\python -m PyInstaller <project.spec>
```

> Full isolation conventions and the per-tool environment list are in
> [环境隔离说明.md](环境隔离说明.md) (Environment Isolation Notes, in Chinese).
> Run `python .tools\check-env-isolation.py` at any time to check the conventions still hold.

For exact entry points and exceptions, always defer to the README inside each subdirectory.

## Data & Result Cautions

- Work on a copy of raw experimental data, and never write outputs into the input directory — avoid overwriting source files.
- TIFF stacks can be very large; reserve enough memory and disk space.
- 8-bit and 16-bit TIFFs have different intensity ranges. Before any quantitative analysis, confirm whether the tool preserves the original bit depth.
- GPA, PPA, DPC, SSB, MLLS and similar results depend on acquisition conditions, calibration and parameter choices. Software output is no substitute for physical validation.

## License & Provenance

This repository is **MIT** (see [LICENSE](LICENSE)), applying to the original code.

**Exception: `03-应变分析/strainpp-GPA应变分析/` is GPL-3.0-or-later** (derived from
[JJPPeters/Strainpp](https://github.com/JJPPeters/Strainpp); that directory carries its own `LICENSE`).
GPL-3.0 is contagious: bundling it with the rest of the code into **a single work** for distribution
requires re-examining the whole under GPL-3.0; merely hosting them side by side in one repository
and distributing them independently is unaffected.
`全整合` (TEM Suite) loads strainpp **in-process via `import`**; if the two are packaged
into a single executable for distribution, verify GPL compliance first.

**Exception: GPL-3.0 third-party components inside released executables.**
The released executables of `05-EELS分析/EELS边缘价态分析工具` and `全整合` both bundle
**`ncempy` (GPL-3.0-or-later)**. Distribution of those executables remains bound by GPL-3.0
for that component, regardless of this project's MIT declaration; the obligations and
workarounds are spelled out in each directory's `THIRD_PARTY_NOTICES.md`.
Additionally, `data/peng_high.json` in `09-HRTEM模拟` and `010-STEM模拟` comes from
**abTEM (GPL-3.0)**.

Per-tool licenses, third-party dependencies, data-file provenance, and third-party material
**deliberately not included** are listed in [NOTICE.md](NOTICE.md).

## README Coverage

READMEs are written per "standalone project or material collection" and list the key files inside.
No README is created for generated files and external dependencies at every level.
