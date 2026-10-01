"""
ptychography.py - Ptychographic iterative engine (ePIE) for 4D-STEM.

Implements:
1. ePIE (extended Ptychographic Iterative Engine, Maiden & Rodenburg 2009)

 约定（与旧版的关键差异）
----
旧版把 k 空间的光阑函数当作实空间探针直接与物体相乘（实/倒空间
混用），且探针更新使用更新前的物体副本。本版约定：

- 探针 P(r) 是实空间函数：由 k 空间光阑 A(k)·exp[iχ(k)] 经
  P = fftshift(ifft2(·)) 得到，包络居中于探测器阵列中心；
- 衍射图强度按"中心束位于数组原点 (0,0)"的 native FFT 约定参与
  匹配（入口处按 center 做圆移）；
- 正/逆传播使用配对的 ifftshift/fftshift，保证 ψ 与探测器幅度在
  同一坐标约定下替换；
- ePIE 更新按原始论文除以 max|P|²、max|O|² 标量分母，且探针更新
  使用本步已更新的物体。

旧版还包含一个冒充 WDD 的"直接法"（BF 孔径平均），已删除；直接
法需求请使用 ssb_core（SSB）或 dpc_core（iDPC）。

绝对标度说明：重建物体的像素尺寸 = 探针步距 / scan_step（像素），
如需物理尺度需结合束流/相机长度标定。
"""
import json
import os

import numpy as np


def _wrap_grid(n):
    """native FFT 约定下的整数频率坐标：[0,1,..,n/2-1,-n/2,..,-1]。"""
    return np.fft.fftfreq(n, 1.0 / n)


def initialize_probe(det_y, det_x, center=None, alpha=8.0, aberrations=None):
    """
    构造实空间探针：k 空间光阑（含像差相位）逆变换到实空间。

    Parameters
    ----------
    det_y, det_x : int - 探测器尺寸
    center : (cy, cx) - 中心束位置（探针本身总是居中；该参数保留
                供调用方传递，仅用于能量核对）
    alpha : float - 光阑半径（探测器像素）
    aberrations : dict - 像差系数（k 空间相位）：C1（离焦）、
                A1/A2（2/3 重像散）、C3（球差），单位：弧度/像素幂

    Returns
    -------
    probe : ndarray (det_y, det_x) complex，包络居中
    """
    ky = _wrap_grid(det_y)[:, None]
    kx = _wrap_grid(det_x)[None, :]
    r = np.sqrt(ky ** 2 + kx ** 2)
    theta = np.arctan2(ky, kx)

    aperture = np.exp(-(r / max(alpha, 1e-6)) ** 10)

    phase = np.zeros((det_y, det_x))
    if aberrations:
        if 'C1' in aberrations:
            phase += aberrations['C1'] * r ** 2
        if 'A1' in aberrations:
            phase += aberrations['A1'] * r ** 2 * np.cos(2 * theta)
        if 'A2' in aberrations:
            phase += aberrations['A2'] * r ** 3 * np.cos(3 * theta)
        if 'C3' in aberrations:
            phase += aberrations['C3'] * r ** 4

    probe_k = aperture * np.exp(1j * phase)
    # native k（DC 在 [0,0]）→ 实空间包络居中
    return np.fft.fftshift(np.fft.ifft2(probe_k))


def epie_reconstruct(datacube, probe_init, center, scan_step=1,
                     n_iterations=50, step_size=0.5, regularization=1e-6,
                     verbose=True, should_stop=None):
    """
    ePIE 同时重建物体透射函数与探针。

    Parameters
    ----------
    datacube : ndarray (scan_y, scan_x, det_y, det_x) 强度
    probe_init : ndarray (det_y, det_x) complex，实空间探针
    center : (cy, cx) 中心束在探测器阵列中的位置（像素）。仅取整数
                round：亚像素残差（<0.5 px）等效为微小探针位置偏差，
                ePIE 的探针更新会吸收它；如需精确亚像素平移应施加
                相位斜坡而非移位整数格点。
    scan_step : float 相邻探针位置在物体阵列上的步距（像素）。
                1 表示物体像素 = 扫描步距（无超分辨）；<1（如 0.5）
                才利用衍射图的高角信息做超分辨重建，代价是物体阵列
                变大、迭代变慢。
    n_iterations, step_size, regularization : 迭代参数
    should_stop : callable，返回 True 时取消。取消时返回当前迭代轮的
                部分重建（迭代法的部分结果仍是合法估计）并置
                cancelled=True；调用方应检查该标志。

    Returns
    -------
    dict：object（复数，覆盖扫描区）、probe、amplitude、phase、
    errors、n_iterations、cancelled
    """
    datacube = np.asarray(datacube)
    scan_y, scan_x, det_y, det_x = datacube.shape
    cy, cx = int(round(center[0])), int(round(center[1]))

    # 中心束移到 native (0,0)；幅度 = sqrt(强度)。保持在输入精度
    # （float32 进 float32 出）并就地开方：旧的 float64 临时副本会让
    # 大立方体的峰值内存翻倍。
    rolled = np.roll(datacube, (-cy, -cx), axis=(2, 3))
    np.maximum(rolled, 0, out=rolled)
    measured = np.sqrt(rolled, out=rolled)

    # 物体：覆盖整个扫描行程 + 一个探针尺寸
    obj_y = int(round((scan_y - 1) * scan_step)) + det_y
    obj_x = int(round((scan_x - 1) * scan_step)) + det_x
    obj = np.ones((obj_y, obj_x), dtype=np.complex128)

    probe = np.array(probe_init, dtype=np.complex128, copy=True)
    # 能量归一：初始 Σ|ψ|² 对齐平均每张 DP 的总计数，加速收敛
    psi0 = probe
    e_probe = float(np.sum(np.abs(psi0) ** 2))
    e_meas = float(np.mean(np.sum(measured ** 2, axis=(2, 3))))
    if e_probe > 0 and e_meas > 0:
        probe *= np.sqrt(e_meas / e_probe)

    hy, hx = det_y // 2, det_x // 2

    # 探针位置（物体坐标，patch 左上角）；scan_step>1 时按步距展开
    positions = []
    for sy in range(scan_y):
        for sx in range(scan_x):
            py = int(round(sy * scan_step)) + hy
            px = int(round(sx * scan_step)) + hx
            y0, x0 = py - hy, px - hx
            if y0 < 0 or x0 < 0 or y0 + det_y > obj_y or x0 + det_x > obj_x:
                continue
            positions.append((sy, sx, y0, x0))
    n_positions = len(positions)

    errors = []
    cancelled = False
    rng = np.random.default_rng(0)

    for iteration in range(n_iterations):
        if should_stop is not None and should_stop():
            cancelled = True
            break
        order = rng.permutation(n_positions)
        total_error = 0.0

        for idx in order:
            sy, sx, y0, x0 = positions[idx]
            obj_patch = obj[y0:y0 + det_y, x0:x0 + det_x]

            # 出射波（居中）→ native k 空间
            psi = probe * obj_patch
            Psi = np.fft.fft2(np.fft.ifftshift(psi))
            D_amp = measured[sy, sx]

            Psi_amp = np.abs(Psi)
            Psi_amp_safe = np.maximum(Psi_amp, 1e-12)
            total_error += float(
                np.sum((Psi_amp - D_amp) ** 2) / (np.sum(D_amp ** 2) + 1e-12))

            # 幅度替换（保相位）→ 回到实空间
            Psi_new = D_amp * Psi / Psi_amp_safe
            psi_new = np.fft.fftshift(np.fft.ifft2(Psi_new))
            dpsi = psi_new - psi

            # ePIE 物体更新：除以本 patch 的 max|P|²
            probe_norm = float(np.max(np.abs(probe) ** 2)) + regularization
            obj[y0:y0 + det_y, x0:x0 + det_x] += \
                step_size * np.conj(probe) / probe_norm * dpsi

            # ePIE 探针更新：使用本步已更新的物体
            obj_upd = obj[y0:y0 + det_y, x0:x0 + det_x]
            obj_norm = float(np.max(np.abs(obj_upd) ** 2)) + regularization
            probe += step_size * np.conj(obj_upd) / obj_norm * dpsi

        errors.append(total_error / max(n_positions, 1))
        if verbose and (iteration + 1) % 10 == 0:
            print(f"  ePIE iter {iteration + 1}/{n_iterations}, "
                  f"error {errors[-1]:.6f}")

    # 裁出扫描覆盖的物体区域（中心样本点对齐）
    out_y = int(round((scan_y - 1) * scan_step)) + 1
    out_x = int(round((scan_x - 1) * scan_step)) + 1
    obj_crop = obj[hy:hy + out_y, hx:hx + out_x]

    return {
        'object': obj_crop,
        'probe': probe,
        'amplitude': np.abs(obj_crop),
        'phase': np.angle(obj_crop),
        'errors': errors,
        'n_iterations': n_iterations,
        'cancelled': cancelled,
    }


def plot_ptychography_results(result, name, output_dir):
    """Visualize ptychography results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    im = axes[0, 0].imshow(result['amplitude'], cmap='gray')
    axes[0, 0].set_title('Object Amplitude |O|')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)

    phase = result['phase']
    vmax = max(abs(phase.min()), abs(phase.max())) or 1.0
    im = axes[0, 1].imshow(phase, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title('Object Phase φ')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], shrink=0.8)

    im = axes[0, 2].imshow(phase, cmap='hsv')
    axes[0, 2].set_title('Phase (HSV)')
    axes[0, 2].axis('off')

    if 'probe' in result:
        probe_amp = np.abs(result['probe'])
        im = axes[0, 3].imshow(probe_amp, cmap='inferno')
        axes[0, 3].set_title('Probe |P|')
        axes[0, 3].axis('off')
        plt.colorbar(im, ax=axes[0, 3], shrink=0.8)
    else:
        axes[0, 3].axis('off')

    if 'probe' in result:
        probe_phase = np.angle(result['probe'])
        im = axes[1, 0].imshow(probe_phase, cmap='RdBu_r')
        axes[1, 0].set_title('Probe Phase')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
    else:
        axes[1, 0].axis('off')

    if 'errors' in result and result['errors']:
        axes[1, 1].semilogy(result['errors'])
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].set_title('Convergence')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].axis('off')

    im = axes[1, 2].imshow(np.real(result['object']), cmap='RdBu_r')
    axes[1, 2].set_title('Re[O]')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], shrink=0.8)

    im = axes[1, 3].imshow(np.imag(result['object']), cmap='RdBu_r')
    axes[1, 3].set_title('Im[O]')
    axes[1, 3].axis('off')
    plt.colorbar(im, ax=axes[1, 3], shrink=0.8)

    plt.suptitle(f'Ptychography (ePIE): {name}', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{name}_ptychography.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def run_ptychography(datacube, center, alpha, method='epie',
                     n_iterations=50, step_size=0.5, scan_step=1.0,
                     name='sample', output_dir=None, verbose=True,
                     should_stop=None):
    """
    Main ptychography function.

    method : 'epie'。旧版 'wdd' 实为 BF 孔径平均（非 WDD），已删除；
    直接法重构请使用 SSB（ssb_core）或 iDPC（dpc_core）。
    scan_step : 相邻探针位置在物体阵列上的步距（像素）。1 = 物体像素
                等于扫描步距；<1 启用超分辨采样（物体更大、更慢）。
    """
    scan_y, scan_x, det_y, det_x = datacube.shape

    if verbose:
        print(f"  Ptychography ({method}): scan={scan_y}x{scan_x}, "
              f"det={det_y}x{det_x}")

    if method != 'epie':
        raise ValueError(
            f"未知方法 '{method}'。可用：'epie'。"
            "直接法重构请用 SSB（ssb_core）或 iDPC（dpc_core）。")

    probe = initialize_probe(det_y, det_x, center=center, alpha=alpha)
    result = epie_reconstruct(
        datacube, probe, center=center,
        scan_step=scan_step,
        n_iterations=n_iterations,
        step_size=step_size,
        verbose=verbose,
        should_stop=should_stop,
    )

    if output_dir:
        plot_ptychography_results(result, name, output_dir)
        np.save(os.path.join(output_dir, f'{name}_ptychography_phase.npy'),
                result['phase'])
        np.save(os.path.join(output_dir, f'{name}_ptychography_amplitude.npy'),
                result['amplitude'])
        with open(os.path.join(output_dir,
                               f'{name}_ptychography_metadata.json'),
                  'w', encoding='utf-8') as fj:
            json.dump({
                'method': method,
                'n_iterations': n_iterations,
                'step_size': step_size,
                'scan_step': scan_step,
                'iterations_completed': len(result['errors']),
                'final_error': (float(result['errors'][-1])
                                if result['errors'] else None),
                'cancelled': bool(result.get('cancelled', False)),
            }, fj, indent=2, ensure_ascii=False)

    return result
