# Security policy

## Supported version

Only v2.1.x is supported. The root-level historical `OPtimize.exe` and its
Python 3.7 runtime are unsupported and must not be distributed or used.

## Threat model

The application is offline and does not execute shell commands, deserialize
pickle/YAML, evaluate user code, or make network requests. Its main untrusted
input is a TIFF file. Relevant risks are malformed-image denial of service,
native codec vulnerabilities, path aliasing, output truncation, and scientific
metadata corruption.

Mitigations include full page preflight, numeric/shape validation, memory
budget checks, exact input/output path comparison, transactional output,
full post-write page decoding, pre/post input hashing, paired TIFF/sidecar
rollback, entity-safe OME-XML parsing, pinned dependencies, hash-locked Windows
wheels, and release-time dependency auditing.

## Reporting

Do not attach private microscopy data to a report. Provide:

- application version and executable SHA-256;
- operating system and available memory;
- TIFF structural metadata with sample data removed;
- the `.stem.json` sidecar if it contains no sensitive paths;
- the relevant rotating log excerpt.

Full input/output paths in logs and file names in sidecars may contain sensitive
project names. Redact them before sharing. Sidecars intentionally omit parent
directory paths.
