# -*- coding: utf-8 -*-
"""
全局常量与配置定义

集中管理应用版本号、UI 配色、默认参数等常量，
避免硬编码分散在各模块中。
"""

# ==================== 应用信息 ====================
APP_TITLE = "电镜晶体/非晶区域统计分析工具"
APP_VERSION = "5.3"
APP_FULL_TITLE = f"{APP_TITLE} v{APP_VERSION}"

# ==================== UI 配色（深色主题） ====================
DARK_CANVAS_BG = '#1e1e1e'
DARK_TEXT_BG = '#1e1e1e'
LIGHT_TEXT_FG = '#cccccc'
ACCENT_GREEN = '#4caf50'
ACCENT_RED = '#f44336'

# ==================== 窗口默认尺寸 ====================
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 900
MIN_WINDOW_WIDTH = 1000
MIN_WINDOW_HEIGHT = 750
RIGHT_PANEL_WIDTH = 360
DEFAULT_CANVAS_WIDTH = 600
DEFAULT_CANVAS_HEIGHT = 500

# ==================== 分割方法 ====================
SEGMENTATION_METHODS = [
    "threshold",    # 全局固定阈值
    "otsu",         # Otsu 自动阈值（新增）
    "adaptive",     # 自适应阈值
    "canny",        # Canny 边缘检测
    "watershed",    # 分水岭算法
]

# 分割方法中文名称映射
METHOD_NAMES_CN = {
    "threshold": "固定阈值",
    "otsu": "Otsu自动阈值",
    "adaptive": "自适应阈值",
    "canny": "Canny边缘",
    "watershed": "分水岭",
}

# ==================== 默认参数 ====================
DEFAULT_THRESHOLD = 128
DEFAULT_ADAPTIVE_BLOCKSIZE = 11
DEFAULT_ADAPTIVE_C = 2
DEFAULT_CANNY_LOW = 50
DEFAULT_CANNY_HIGH = 150
DEFAULT_WATERSHED_KERNEL = 3
DEFAULT_PIXEL_SIZE_NM = 1.0       # nm/像素（假定值；未标定时实际单位是 px）
DEFAULT_PIXEL_CALIBRATED = False  # 像素尺寸是否经过显微镜标定（未标定禁止按 nm² 引用面积）
DEFAULT_TIME_INTERVAL = 1.0       # 时间间隔
DEFAULT_CANNY_MIN_AREA = 50       # Canny 最小轮廓面积（像素）

# ==================== 前景极性 ====================
# "晶体/前景"由灰度阈值定义：亮区或暗区。HAADF Z 衬度通常亮区为晶体；
# TEM 明场衍射衬度下晶体区域常更暗，需要反相。
FOREGROUND_BRIGHT = "bright"
FOREGROUND_DARK = "dark"
FOREGROUND_OPTIONS = {
    FOREGROUND_BRIGHT: "亮区为晶体（HAADF/暗场，默认）",
    FOREGROUND_DARK: "暗区为晶体（TEM 明场衍射衬度）",
}

# ==================== 图像显示 ====================
# uint16 -> uint8 的缩放不再使用固定 /256：12-bit 相机数据（0-4095）会被
# 压到 0-15，固定阈值与 Otsu 全部失效。现按数据实际量程选择右移位数
# （0/4/8 位），见 core.segmentation.uint16_to_uint8。
MIN_REGION_PIXELS = 3             # 连通区域计入"区域数量"的最小像素数（与轮廓统计同口径）
MAX_ZOOM = 5.0
MIN_ZOOM = 0.05
ZOOM_FACTOR = 1.25
RESIZE_DEBOUNCE_MS = 150          # 窗口 resize 防抖延迟

# ==================== 标注颜色 (BGR) ====================
COLOR_CRYSTALLINE = [0, 255, 0]   # 晶体区域 - 绿色
COLOR_AMORPHOUS = [0, 0, 255]     # 非晶区域 - 红色
AMORPHOUS_OVERLAY_ALPHA = 0.35    # 非晶区域叠加透明度
CRYSTALLINE_OVERLAY_ALPHA = 0.4   # 晶体区域叠加透明度

# ==================== 输出文件 ====================
ANNOTATED_SUBDIR = "annotated_tif"
SUPPORTED_EXTENSIONS = ('.tif', '.tiff')

# ==================== 形状因子参考值 ====================
SHAPE_FACTOR_CIRCLE = 1.0         # 完美圆形
SHAPE_FACTOR_SQUARE = 0.7854      # 正方形
SHAPE_FACTOR_RECT_2_1 = 0.6981    # 长方形(2:1)

# ==================== 结果表列名（schema 常量） ====================
# 结果 DataFrame/CSV 的列名是跨模块契约（measurement 生成、analysis/exporter
# 消费），必须集中定义；散落字符串会在改名时静默断链。
COL_FILENAME = '文件名'
COL_SLICE = '切片'
COL_TOTAL_PIXELS = '总像素数'
COL_CRYST_PIXELS = '晶体区域像素数'
COL_AMORPH_PIXELS = '非晶区域像素数'
COL_CRYST_AREA = '晶体区域实际面积(nm²)'
COL_AMORPH_AREA = '非晶区域实际面积(nm²)'
COL_CRYST_PERIM = '晶体区域周长(nm)'
COL_N_REGIONS = '晶体区域数量'
COL_SF_MEAN = '形状因子_平均值'
COL_SF_STD = '形状因子_标准差'
COL_SF_MIN = '形状因子_最小值'
COL_SF_MAX = '形状因子_最大值'
COL_CRYST_RATIO = '晶体区域面积比例'
COL_AMORPH_RATIO = '非晶区域面积比例'
COL_PIXEL_CALIB = '像素尺寸标定'
COL_OTSU = 'Otsu阈值'
COL_GROWTH_RATE = '径向生长速率(nm/单位时间)'

# ==================== 逐区域导出 ====================
REGION_EXCEL_MAX_ROWS = 200_000   # 逐区域数据写入 Excel 的行数上限（超出仅写 CSV）

# ==================== 阈值常量（原散落的魔法数字） ====================
# 区域形状一致性评估分档（切片内形状因子标准差的均值）
SHAPE_UNIFORMITY_TIERS = (
    (0.05, "区域形状高度一致"),
    (0.10, "区域形状较一致"),
    (0.20, "区域形状较分散"),
)
SHAPE_UNIFORMITY_WORST = "区域形状显著分散"

# "验证计算"面板的整体形状因子判读分档
VERIFY_SF_NEAR_CIRCLE = 0.9   # > 0.9 判为接近圆形
VERIFY_SF_REGULAR = 0.7       # > 0.7 判为较规则

# 预览整卷载入内存的确认阈值（文件超过该大小先询问用户）
PREVIEW_LARGE_FILE_MB = 2048
