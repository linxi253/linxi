# Third-party notices

## Bundled third-party components

The v5 application depends on the packages listed in `requirements.lock.txt`
(PyQt/PySide, numpy, tifffile, imagecodecs, scikit-image, matplotlib and their
transitive dependencies). Each is distributed under its own licence; consult
the package metadata for the exact terms.

## Legacy material excluded from this repository

The current v5 Python application does not load `PASAD-tools.dll`,
`PASAD-tools.gt1`, `Radial_Profile_Angle.jar`, or the bundled video. They are
historical material and are excluded from the v5 build.

Before redistributing any legacy file, verify its original source, licence, and
hash independently. In particular, `PASAD-tools.dll` was not Authenticode
signed when this project was repaired.

The DigitalMicrograph scripts (`HRTEM Filter.s`, `去噪音插件.s`/`.txt`) and the
ImageJ plugin `Radial_Profile_Angle.jar` were authored by third parties and
their redistribution terms were never confirmed. They are therefore **not
included in this repository** — see `legacy/README.md` for details and original
sources.

`legacy/imagej/FFT for image stack.ijm` is retained. It carries no author or
copyright header and consists of generic ImageJ batch commands; its provenance
is unconfirmed, so treat it as reference material rather than a redistributable
component.
