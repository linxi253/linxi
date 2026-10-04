# AIforTEM 仓库约定（源码）

本仓库收录 **20 个工具项目**，同一个 Git 仓库，但**每个项目有独立的解释器、锁和 venv**。
项目清单（Python 版本、母本、锁链、测试入口）见 [.tools/projects.json](.tools/projects.json)；
环境契约与重建方法见 [环境隔离说明](环境隔离说明.md)。

## 导航

各项目的源码入口、职责与验证方式见该项目自己的 `AGENTS.md` / `DEVELOPMENT.md`。
分类总览见 [README.md](README.md)。

## 规则

1. **一个项目一个环境**：不要把项目依赖装进 conda base，也不要用 PATH 上的裸
   `python` 运行项目；用项目自己 `.venv` 里的解释器。
2. **锁是安装依据**：按 `.tools/projects.json` 的 `install.chain` 安装；
   不要用 `pip install -r requirements.txt` 代替锁。
3. **原始实验数据只读**：不要修改、移动或删除采集数据与历史归档。
4. **保留已有改动**：进入任何项目前先看 `git status`；不要丢弃他人未提交的修改。
5. **按影响测试**：改哪个项目就跑哪个项目的测试；纯文档改动检查路径与关键符号即可。
   环境缺失时报告「未验证」，不要为了跑通而改环境或降级锁。
6. **不要把「未运行」写成「通过」**：区分通过 / 跳过 / 环境缺失 / 未运行。

## 验证

从**仓库根**用第 2 节建立的**工具解释器**运行（不要用 PATH 上的裸 `python`）：

```powershell
& $toolPython .tools/verify-envs.py          # 只验证，不安装
& $toolPython .tools/check-env-isolation.py  # 隔离与依赖闭包检查
```

工具环境的建立与 `$toolPython` 的定义见 [环境隔离说明](环境隔离说明.md) 第 2 节。
