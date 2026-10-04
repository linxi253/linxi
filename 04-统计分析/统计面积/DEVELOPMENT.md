# 统计面积 源码与验证导航

TIFF 堆栈上的手工多边形、标尺换算与可恢复面积测量。

**状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[统计面积.py](统计面积.py)。
- 版本来源：未登记统一静态版本源；按 README、源码和构建记录分别核实，不能以旧 EXE 文件名推断。
- 锁文件：[requirements.lock.txt](requirements.lock.txt)
- 构建配置：[统计面积v2.spec](统计面积v2.spec)

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| GUI、加载代次和项目交互 | [统计面积.py](统计面积.py) | TiffStackViewer |
| 几何计算、坐标和排序 | [area_core.py](area_core.py) | polygon_area、polygon_perimeter、flip_polygon_y、natural_sort_key |

## 保持的约束

- 面积与周长依赖像素标尺及单位；显示坐标变换不能改变导出几何口径。
- 项目加载要验证标量和顶点，异常项按现有警告/降级处理；不静默删除越界原始测量。
- 保持加载代次、取消和未保存修改提示；显示灰度拉伸不等于原始数据变化。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python >=3.10，本机母本 3.10）。
- **安装锁**：`requirements.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`tifffile`、`numpy`、`pandas`、`matplotlib`、`PIL`、`openpyxl`。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
