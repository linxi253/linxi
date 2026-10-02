# 贡献说明（CONTRIBUTING）

## 各工具 README 的「速查」区块（README-QUICKREF）

各工具 README 中由 `<!-- README-QUICKREF:BEGIN … -->` / `<!-- README-QUICKREF:END -->`
包住的「速查」区块为**手工维护**，没有生成脚本（历史上的 `tools/gen-readme-block.py`
已不在仓库中）。维护约定：

- 「当前版本」必须与该块「版本来源」行指向的单一来源一致——通常是项目内
  `pyproject.toml` 的 `version`，或代码中的 `__version__` 常量；以「版本来源」行的
  声明为准。
- 改版本号时：先改版本来源处的常量，再同步本速查块与 `CHANGELOG.md`，三处一起提交。
- 「许可证」「依赖锁定」「入口」等行同样以仓库内实际文件为准；引用的文件路径必须真实存在。
- BEGIN/END 标记行本身请保留，便于工具定位与审计。
