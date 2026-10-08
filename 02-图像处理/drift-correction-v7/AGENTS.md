# drift-correction-v7 开发约定

二维灰度 TIFF 堆栈的纯平移漂移检测与矫正。 **状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。

按任务查看 [源码与验证导航](DEVELOPMENT.md)，使用与科学说明见 [README](README.md)。

## 修改边界

- 保留已有修改与未跟踪文件；可读性整理只补导航，不改算法、路径、输出或依赖。
- 使用 .venv/Scripts/python.exe；锁文件与测试前置条件见开发说明，缺失时记录未验证，不回退全局 Python。
- 原始数据、发布成品、权重及历史验证材料保持原样。只读导航无需安装、重打包或执行业务模块。

## 关键行为

- 保持纯平移模型；不能将 ImageJ/Fiji 参数对齐误写成支持相同的旋转/刚体模型。
- 输出不得覆盖输入，取消/异常只清理本次临时文件；原子替换、TIFF 校验和共同有效区裁剪都要保留。
- 逐帧质量、插值段、dx/dy 方向、边界模式与审计字段影响结果解释。质量不足不能静默伪装成功。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python 3.10–3.13，本机母本 3.10）。
- **安装锁**：`requirements.resolved-win-py310.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`cv2`、`matplotlib`、`tifffile`、`tkinterdnd2`。
- **注意**：requirements.lock 头部标注按 Windows x64 / Python 3.10.9 验证。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
