#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""引擎同步守门：09-HRTEM模拟/tem_sim <-> 010-STEM模拟/stem_sim（审计条目 15 轻量路线）。

两个包是同一套模拟引擎的两份拷贝（审计权威清单 条目 15），历史已因单侧打补丁
分叉——相位标度 /pixel_area 曾修在 010、漏在 09，属真实发生过的静默数值错误。
在抽出单一真源（共享 emsim 包）之前，本脚本充当守门检查：

  1. 对两侧公共模块（constants/scattering/structure/multislice/microscope）
     做 AST 级函数体比对：同名函数的 AST（剥除 docstring 后）不一致即失败；
  2. 对两侧 data/ 下的数据文件做 sha256 比对，不一致即失败；
  3. 存在未豁免的不一致时以非零码退出，把「只改一侧」变成 CI 可检出的红灯。

纯标准库实现，无第三方依赖；兼容 Python 3.7+（本地与 CI 均可直接运行）。
用法：python .tools/check_engine_sync.py
"""
import ast
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 比对对象：两个同源引擎包与其公共模块清单（审计条目 15 「位置」小节所列）。
SIDES = {
    "09": REPO_ROOT / "09-HRTEM模拟" / "tem_sim",
    "010": REPO_ROOT / "010-STEM模拟" / "stem_sim",
}
COMMON_MODULES = [
    "constants.py",
    "scattering.py",
    "structure.py",
    "multislice.py",
    "microscope.py",
]

# ---------------------------------------------------------------------------
# 豁免表：qualified 函数名（"模块.py:函数名"，类方法记作 "模块.py:类.方法"）→ 理由。
#
# 维护纪律：新增豁免必须写明理由并经评审；豁免只允许覆盖「已确认无害」的差异，
# 不得为了让本脚本变绿而添加——本守门的存在意义就是把单侧修改暴露成红灯。
#
# 「docstring 差异」不进此表：函数体哈希前统一剥除 docstring（两侧把各自的修复
# 说明写进自家 docstring/注释属预期行为），Python 注释同样不进入 AST、天然不参与。
# ---------------------------------------------------------------------------
EXEMPT = {
    # 注：工单表述为「get_factors 的 scalar 返回分支」，但该分支实际位于
    # electron_scattering_factor（09 侧 :176/:182），get_factors 两侧逐字一致。
    "scattering.py:electron_scattering_factor":
        "09 侧多一个 scalar 返回分支（scalar = np.ndim(g)==0；"
        "return float(fe) if scalar else fe），010 侧恒 return fe"
        "——纯 API 便利差异，无物理含义（审计条目 15 更正(3)）",
}


def _is_str_literal(node):
    """docstring 判定：首语句是否为字符串字面量（兼容 3.7 的 ast.Str 与 3.8+ 的 ast.Constant）。"""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    try:
        return isinstance(node, ast.Str)  # Python <3.8 的 parse 产物
    except AttributeError:  # pragma: no cover - 未来版本移除 ast.Str 时兜底
        return False


class _DocstringStripper(ast.NodeTransformer):
    """剥除模块/类/函数的 docstring，使哈希只反映代码结构。"""

    def _strip(self, node):
        self.generic_visit(node)
        body = node.body
        if body and isinstance(body[0], ast.Expr) and _is_str_literal(body[0].value):
            body = body[1:]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not body:
            body = [ast.Pass()]
        node.body = body
        return node

    visit_Module = _strip
    visit_ClassDef = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip


def parse_tree(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _DocstringStripper().visit(tree)


def function_index(tree):
    """收集全部函数（含类方法）→ {qualified 名: 剥除 docstring 后的 ast.dump}。"""
    out = {}

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qname = prefix + child.name
                out[qname] = ast.dump(child)
                walk(child, qname + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")

    walk(tree, "")
    return out


def class_index(tree):
    """顶层类名集合（用于「仅单侧存在」提示）。"""
    return {n.name for n in tree.body if isinstance(n, ast.ClassDef)}


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover - 极老解释器兜底
        pass

    failures = []   # 未豁免的不一致 → 判失败
    exempted = []   # 豁免表命中的差异 → 放行但展示
    notes = []      # 仅单侧存在（新增/独有成员）→ 提示，不计失败

    print("== 引擎同步守门：%s <-> %s ==" % (SIDES["09"].name, SIDES["010"].name))

    # -- 1) 公共模块的 AST 级函数体比对 --------------------------------------
    for module in COMMON_MODULES:
        paths = {side: pkg / module for side, pkg in SIDES.items()}
        missing = [side for side, p in paths.items() if not p.is_file()]
        if missing:
            failures.append("%s: 文件在 %s 侧缺失" % (module, "/".join(missing)))
            continue

        trees = {side: parse_tree(p) for side, p in paths.items()}
        funcs = {side: function_index(t) for side, t in trees.items()}
        classes = {side: class_index(t) for side, t in trees.items()}

        for name in sorted(set(funcs["09"]) | set(funcs["010"])):
            key = "%s:%s" % (module, name)
            in09, in010 = name in funcs["09"], name in funcs["010"]
            if in09 and in010:
                if funcs["09"][name] != funcs["010"][name]:
                    if key in EXEMPT:
                        exempted.append("%s — %s" % (key, EXEMPT[key]))
                    else:
                        failures.append(key)
            else:
                side = "09" if in09 else "010"
                notes.append("%s  %s" % (key, "(仅 %s 侧)" % side))

        for cname in sorted(set(classes["09"]) ^ set(classes["010"])):
            side = "09" if cname in classes["09"] else "010"
            notes.append("%s:%s  (仅 %s 侧)" % (module, cname, side))

    # -- 2) data/ 数据文件 sha256 比对 ---------------------------------------
    data_dirs = {side: pkg / "data" for side, pkg in SIDES.items()}
    data_dirs_ok = all(d.is_dir() for d in data_dirs.values())
    for side, d in data_dirs.items():
        if not d.is_dir():
            failures.append("data/: 目录在 %s 侧缺失" % side)
    if data_dirs_ok:
        names09 = {p.name for p in data_dirs["09"].iterdir() if p.is_file()}
        names010 = {p.name for p in data_dirs["010"].iterdir() if p.is_file()}
        for name in sorted(names09 | names010):
            if name not in names09 or name not in names010:
                side = "09" if name in names09 else "010"
                notes.append("data/%s  (仅 %s 侧)" % (name, side))
                continue
            h09 = sha256_of(data_dirs["09"] / name)
            h010 = sha256_of(data_dirs["010"] / name)
            if h09 == h010:
                print("  数据一致  data/%s  sha256=%s…" % (name, h09[:16]))
            else:
                failures.append("data/%s: sha256 不一致（%s… vs %s…）"
                                % (name, h09[:16], h010[:16]))

    # -- 3) 汇总 --------------------------------------------------------------
    if exempted:
        print("\n[豁免放行] %d 处（见脚本头部 EXEMPT 表）" % len(exempted))
        for line in exempted:
            print("  EXEMPT  %s" % line)
    if notes:
        print("\n[仅单侧存在，不计失败，请人工确认是否有意为之] %d 处" % len(notes))
        for line in notes:
            print("  NOTE    %s" % line)
    if failures:
        print("\n[未豁免的不一致] %d 处" % len(failures))
        for line in failures:
            print("  DIFF    %s" % line)
        print("\n结果: FAIL — 公共引擎已再次单侧分叉。请把改动同步到两侧，"
              "或确认无害后在 .tools/check_engine_sync.py 的 EXEMPT 表声明理由。")
        sys.exit(1)
    print("\n结果: OK — 公共模块函数体与数据文件两侧一致（豁免 %d 处）。"
          % len(exempted))
    sys.exit(0)


if __name__ == "__main__":
    main()
