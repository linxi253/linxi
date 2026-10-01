# Windows x64 便携发行版

## 支持范围

发行包面向 Windows 10/11 x86-64。目标电脑无需安装 Python、项目依赖或 CUDA，也无需管理员权限。Windows EXE 不能直接运行在 macOS、Linux 或 Windows ARM64；这些平台必须分别在对应系统上重新构建。PyInstaller 官方文档同样要求针对每个操作系统单独打包。

## 构建

```powershell
.\scripts\build_windows_annotator.ps1
```

构建使用锁定的 `pyinstaller==6.22.2`，执行顺序如下：

1. 构建 one-folder 便携目录；
2. 运行内置 TIFF、Tk、项目、JSON、PNG 和 YOLO 自检；
3. 生成带标注员说明的 ZIP；
4. 构建并自检 one-file 备用 EXE；
5. 为可分发文件生成 SHA-256。

官方建议先验证 one-folder，再尝试 one-file；one-file 每次启动会先释放依赖到临时目录，因此启动更慢。日常分发应优先使用 ZIP 便携版。[PyInstaller operating mode](https://pyinstaller.org/en/stable/operating-mode.html)

## 标注任务分发

负责人可先运行程序，选择 HAADF-STEM 或 HRTEM，再选择保存位置。程序会自动
创建带时间戳的任务文件夹；在主界面按“① 打开图像文件夹”“② 扫描新增图像”
两步加入原图：

```text
task_name/
├── annotation_project.json
├── images/
│   └── <sample_id>/<acquisition_id>/*.tif
├── labels/
└── manifest.csv
```

整个任务文件夹内部使用相对路径，可复制到其他盘符或电脑。建议按 `acquisition_id` 把互不重叠的任务分给不同标注人员。回收时要求对方返回整个任务文件夹，至少必须保留项目文件、标签和 manifest。

标注人员启动软件后只需点击“打开收到的标注任务文件夹”，选择上述整个目录；
不需要识别或手动寻找 JSON 文件。启动页会根据系统 DPI 自动调整尺寸，主界面
右侧控制区可以滚动，以兼容高显示缩放和较小屏幕。

## 安全与签名

内部版本尚未使用商业代码签名证书，因此 Windows SmartScreen 可能显示“未知发布者”。分发时应同时发送 SHA-256 清单，并通过可信渠道让标注人员核对。若要面向组织外大规模分发，应购买受信任的代码签名证书并在构建后签名；自签名证书不会自动获得其他电脑信任。

## 可重现性

便携目录包含 `BUILD_INFO.txt`、`SELF_TEST.json`、标注员说明和第三方组件声明。构建产物不进入 Git；源代码、构建脚本、版本资源与依赖锁定文件进入 Git。
