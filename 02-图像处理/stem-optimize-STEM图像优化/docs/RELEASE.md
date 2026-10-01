# Release procedure

1. Use 64-bit CPython 3.10 on Windows.
2. Run `scripts/build_release.ps1`.
3. Require Ruff, unit tests, pip-audit, dependency consistency, and the
   packaged `--self-test` LZW/OME transaction to pass.
4. Verify Windows ProductVersion/FileVersion and the application startup log.
5. If a trusted code-signing certificate is available, pass its thumbprint and
   verify `Get-AuthenticodeSignature` reports `Valid`.
6. Publish the executable together with `SHA256SUMS.txt`, README, CHANGELOG,
   SECURITY policy, and the exact lock file.
7. Never publish `legacy/`, `build/`, `.venv-build/`, or extracted executables.
