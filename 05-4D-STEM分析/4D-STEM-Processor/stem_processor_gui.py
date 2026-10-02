"""
4D-STEM Processor - One-click GUI application for 4D-STEM data analysis.

Performs DPC/iDPC, SSB, Strain Mapping, Orientation Mapping, Peak Pairs and
Ptychography on DM4 4D-STEM datasets.

Refactored (v2.0):
- all algorithms delegate to ``core/`` (no duplicated implementations);
- the DM4 dtype is read from the file header instead of guessed from the
  file size;
- the DPC/iDPC and SSB checkboxes now actually control the pipeline;
- each file is extracted exactly once and reused by every analysis module;
- logging is thread-safe (worker -> queue -> main-thread polling);
- Stop is cooperative and also interrupts the heavy core loops.
"""
import json
import os
import queue
import sys
import threading
import time
import traceback

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# All figures are rendered to files from the worker thread, so Agg is the
# only safe backend. (TkAgg was previously selected here and then silently
# overridden to Agg by the core module imports.)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Make the project root importable when running from any directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import dm4_io
from core import pipeline
from core.dpc_core import (compute_com_robust, idpc_reconstruct,
                           compute_virtual_images)
from core.ssb_core import ssb_reconstruct
from core.strain_mapping import run_strain_mapping
from core.orientation_mapping import run_orientation_mapping, STRUCTURE_PRESETS
from core.peak_pairs import peak_pairs_mapping, plot_peak_pairs_results
from core.ptychography import run_ptychography

from core import __version__


# ============================================================
# Plotting helpers (presentation only; numerics live in core/)
# ============================================================

def _vmax(arr):
    """Symmetric color scale limit."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 1.0
    return max(abs(float(arr.min())), abs(float(arr.max())), 1e-12)


def _imshow(ax, arr, cmap='gray', **kwargs):
    im = ax.imshow(arr, cmap=cmap, **kwargs)
    ax.axis('off')
    return im


def _save_dpc_plots(name, output_dir, bf, adf, abf, com, phase):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    _imshow(axes[0, 0], bf).axes.set_title('Virtual BF')
    _imshow(axes[0, 1], adf).axes.set_title('Virtual ADF')
    _imshow(axes[0, 2], abf).axes.set_title('Virtual ABF')
    _imshow(axes[0, 3], com['com_magnitude'], cmap='inferno').axes.set_title('|CoM|')

    for ax, arr, title in ((axes[1, 0], com['com_y'], 'CoM-Y'),
                           (axes[1, 1], com['com_x'], 'CoM-X')):
        v = _vmax(arr)
        im = _imshow(ax, arr, cmap='RdBu_r', vmin=-v, vmax=v)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

    v = _vmax(phase)
    im = _imshow(axes[1, 2], phase, cmap='RdBu_r', vmin=-v, vmax=v)
    axes[1, 2].set_title('iDPC Phase')
    plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
    _imshow(axes[1, 3], phase, cmap='viridis').axes.set_title('iDPC Phase')

    plt.suptitle(f'DPC: {name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'{name}_dpc.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def _save_ssb_plots(name, output_dir, ssb):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    _imshow(axes[0, 0], ssb['amplitude']).axes.set_title('SSB Amplitude')
    _imshow(axes[0, 1], ssb['phase'], cmap='viridis').axes.set_title('SSB Phase')
    v = _vmax(ssb['phase'])
    im = _imshow(axes[0, 2], ssb['phase'], cmap='RdBu_r', vmin=-v, vmax=v)
    axes[0, 2].set_title('SSB Phase (+/-)')
    plt.colorbar(im, ax=axes[0, 2], shrink=0.8)
    _imshow(axes[1, 0], np.real(ssb['complex_obj']), cmap='RdBu_r').axes.set_title('Re[psi]')
    _imshow(axes[1, 1], np.imag(ssb['complex_obj']), cmap='RdBu_r').axes.set_title('Im[psi]')
    _imshow(axes[1, 2], ssb['aperture']).axes.set_title('Aperture')
    plt.suptitle(f'SSB: {name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'{name}_ssb.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def _save_comparison_plot(name, output_dir, adf, com, phase, ssb, corr):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    _imshow(axes[0, 0], adf).axes.set_title('ADF')
    _imshow(axes[0, 1], com['com_magnitude'], cmap='inferno').axes.set_title('|CoM|')
    v = _vmax(phase)
    im = _imshow(axes[0, 2], phase, cmap='RdBu_r', vmin=-v, vmax=v)
    axes[0, 2].set_title('iDPC Phase')
    plt.colorbar(im, ax=axes[0, 2], shrink=0.8)
    _imshow(axes[0, 3], com['bf_intensity']).axes.set_title('BF')

    _imshow(axes[1, 0], ssb['amplitude']).axes.set_title('SSB Amp')
    v = _vmax(ssb['phase'])
    im = _imshow(axes[1, 1], ssb['phase'], cmap='RdBu_r', vmin=-v, vmax=v)
    axes[1, 1].set_title('SSB Phase')
    plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
    diff = phase - ssb['phase']
    v = _vmax(diff)
    im = _imshow(axes[1, 2], diff, cmap='RdBu_r', vmin=-v, vmax=v)
    axes[1, 2].set_title('DPC - SSB')
    plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
    axes[1, 3].scatter(phase.ravel()[::5], ssb['phase'].ravel()[::5],
                       s=1, alpha=0.3)
    axes[1, 3].set_xlabel('DPC')
    axes[1, 3].set_ylabel('SSB')
    axes[1, 3].set_title(f'Corr: {corr:.4f}' if corr is not None else 'Corr: n/a')
    plt.suptitle(f'DPC vs SSB: {name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'{name}_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def _save_efield_plot(name, output_dir, com, scale_nm):
    """Render the electric-field / charge-density figure (numeric arrays
    are persisted by process_dm4_file, not by this plotting helper)."""
    com_x, com_y = com['com_x'], com['com_y']
    e_mag = np.sqrt(com_x ** 2 + com_y ** 2)
    charge = (np.gradient(com_x, axis=1) + np.gradient(com_y, axis=0)) / scale_nm

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, arr, title in ((axes[0, 0], com_x, 'E_x'),
                           (axes[0, 1], com_y, 'E_y')):
        v = _vmax(arr)
        im = _imshow(ax, arr, cmap='RdBu_r', vmin=-v, vmax=v)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
    _imshow(axes[0, 2], e_mag, cmap='inferno').axes.set_title('|E|')
    v = _vmax(charge)
    im = _imshow(axes[1, 0], charge, cmap='RdBu_r', vmin=-v, vmax=v)
    axes[1, 0].set_title('Charge Density')
    plt.colorbar(im, ax=axes[1, 0], shrink=0.8)

    step = max(1, com_x.shape[0] // 20)
    y, x = np.mgrid[0:com_x.shape[0]:step, 0:com_x.shape[1]:step]
    axes[1, 1].quiver(x, y, com_x[::step, ::step], -com_y[::step, ::step],
                      e_mag[::step, ::step], cmap='viridis', alpha=0.8)
    axes[1, 1].set_title('E Vectors')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].invert_yaxis()
    axes[1, 2].hist(e_mag.ravel(), bins=100, color='blue', alpha=0.7)
    axes[1, 2].set_title('|E| Distribution')
    plt.suptitle(f'Electric Field: {name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'{name}_efield.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ============================================================
# Per-file processing pipeline
# ============================================================

def process_dm4_file(dm4_path, output_dir, scan_crop, opts, log, should_stop,
                     name=None):
    """Process one DM4 file and return a reusable per-file workspace.

    ``name`` optionally overrides the output basename (used by the GUI to
    disambiguate DM4 files that share a basename in different folders).
    """
    ds = pipeline.prepare_dataset(dm4_path, scan_crop=scan_crop, log=log,
                                  name=name)
    name = ds['name']
    meta = ds['meta']
    data = ds['data']
    center = ds['center']
    alpha = ds['alpha']
    dim_info = ds['dim_info']
    os.makedirs(output_dir, exist_ok=True)
    scan_y, scan_x, det_y, det_x = data.shape
    t_file0 = time.time()

    log("  Computing CoM...")
    com = compute_com_robust(data, center=center, radius=alpha,
                             should_stop=should_stop)
    if should_stop() or com.get('cancelled'):
        return None
    log(f"  CoM-Y: [{com['com_y'].min():.3f}, {com['com_y'].max():.3f}]")

    vimg = compute_virtual_images(data, center, alpha)

    phase = None
    dpc_regularization = opts.get('dpc_regularization', 1e-3)
    if opts.get('dpc'):
        log("  iDPC reconstruction...")
        phase = idpc_reconstruct(com, regularization=dpc_regularization)
        log(f"  Phase: [{phase.min():.3f}, {phase.max():.3f}]")

    ssb = None
    ssb_defocus = opts.get('ssb_defocus', 0.0)
    if opts.get('ssb'):
        log("  SSB reconstruction...")
        ssb = ssb_reconstruct(data, alpha_pixels=alpha, center=center,
                              defocus_rad=ssb_defocus,
                              verbose=False, should_stop=should_stop)
        if should_stop() or ssb.get('cancelled'):
            return None
        log(f"  SSB phase: [{ssb['phase'].min():.3f}, "
            f"{ssb['phase'].max():.3f}]")

    corr = None
    if phase is not None and ssb is not None and phase.shape == ssb['phase'].shape:
        corr = float(np.corrcoef(phase.ravel(), ssb['phase'].ravel())[0, 1])
        log(f"  DPC-SSB correlation: {corr:.4f}")

    # ---- save numeric results (only the modules that were enabled) ----
    if phase is not None:
        np.save(os.path.join(output_dir, f'{name}_dpc_phase.npy'), phase)
    if ssb is not None:
        np.save(os.path.join(output_dir, f'{name}_ssb_phase.npy'),
                ssb['phase'])
        np.save(os.path.join(output_dir, f'{name}_ssb_amplitude.npy'),
                ssb['amplitude'])
        np.save(os.path.join(output_dir, f'{name}_ssb_object.npy'),
                ssb['complex_obj'])
    np.save(os.path.join(output_dir, f'{name}_bf.npy'), vimg['bf'])
    np.save(os.path.join(output_dir, f'{name}_adf.npy'), vimg['adf'])
    np.save(os.path.join(output_dir, f'{name}_abf.npy'), vimg['abf'])
    np.save(os.path.join(output_dir, f'{name}_com_x.npy'), com['com_x'])
    np.save(os.path.join(output_dir, f'{name}_com_y.npy'), com['com_y'])
    np.save(os.path.join(output_dir, f'{name}_com_magnitude.npy'),
            com['com_magnitude'])
    np.save(os.path.join(output_dir, f'{name}_bf_intensity.npy'),
            com['bf_intensity'])

    scale_nm = ds['scale_nm']
    charge = (np.gradient(com['com_x'], axis=1)
              + np.gradient(com['com_y'], axis=0)) / scale_nm
    np.save(os.path.join(output_dir, f'{name}_charge_density.npy'), charge)

    # ---- plots ----
    log("  Generating plots...")
    if phase is not None:
        _save_dpc_plots(name, output_dir, vimg['bf'], vimg['adf'],
                        vimg['abf'], com, phase)
    if ssb is not None:
        _save_ssb_plots(name, output_dir, ssb)
    if phase is not None and ssb is not None:
        _save_comparison_plot(name, output_dir, vimg['adf'], com, phase,
                              ssb, corr)
    _save_efield_plot(name, output_dir, com, scale_nm)

    dim_scores = {
        'score_scan_first': dim_info.get('score_scan_first'),
        'score_det_first': dim_info.get('score_det_first'),
    }
    try:
        file_size_gb = round(os.path.getsize(dm4_path) / 1024 ** 3, 3)
    except OSError:
        file_size_gb = None
    result_meta = {
        'name': name,
        'file': os.path.basename(dm4_path),
        'source_path': dm4_path,
        'file_size_gb': file_size_gb,
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'processor_version': __version__,
        'scan_shape': [scan_y, scan_x],
        'det_shape': [det_y, det_x],
        'dtype': meta['dtype_name'],
        'center': [float(center[0]), float(center[1])],
        'alpha': int(alpha),
        'scan_step_nm': float(scale_nm),
        'dm4_scales': meta.get('scales'),
        'dm4_units': meta.get('units'),
        'preprocess': {'pmin': 1, 'pmax': None, 'dtype': 'float32'},
        'dpc_regularization': float(dpc_regularization),
        'ssb_defocus_rad': float(ssb_defocus),
        'dimension_swapped': bool(dim_info.get('swapped')),
        'dimension_reason': dim_info.get('reason'),
        'dimension_scores': {k: (float(v) if v is not None else None)
                             for k, v in dim_scores.items()},
        'dpc_ssb_correlation': corr,
        'dpc_phase_range': ([float(phase.min()), float(phase.max())]
                            if phase is not None else None),
        'ssb_phase_range': ([float(ssb['phase'].min()),
                             float(ssb['phase'].max())]
                            if ssb is not None else None),
        'com_y_range': [float(com['com_y'].min()), float(com['com_y'].max())],
        'com_x_range': [float(com['com_x'].min()), float(com['com_x'].max())],
        'elapsed_sec': round(time.time() - t_file0, 1),
        'options': dict(opts),
    }
    with open(os.path.join(output_dir, f'{name}_metadata.json'),
              'w', encoding='utf-8') as fj:
        json.dump(result_meta, fj, indent=2, ensure_ascii=False)

    log(f"  Done in {result_meta['elapsed_sec']}s - results saved to: "
        f"{output_dir}")
    return {
        'meta': result_meta,
        'data': data,
        'center': center,
        'alpha': alpha,
        'com': com,
        'vimg': vimg,
        'phase': phase,
        'ssb': ssb,
        'name': name,
        'output_dir': output_dir,
        'scale_nm': scale_nm,
    }


# ============================================================
# GUI application
# ============================================================

class STEMProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("4D-STEM Processor - Full Analysis Suite")
        self.root.geometry("950x800")
        self.root.minsize(850, 700)

        self.processing = False
        self.results = []
        self.msg_queue = queue.Queue()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, self._poll_messages)

    def _on_close(self):
        if self.processing:
            if not messagebox.askyesno(
                    "Confirm exit",
                    "Processing is still running.\n"
                    "Results for the current file may be incomplete.\n"
                    "Quit anyway?"):
                return
            self.processing = False
        self.root.destroy()

    # ---- UI ----
    def _build_ui(self):
        title_frame = ttk.Frame(self.root, padding=10)
        title_frame.pack(fill=tk.X)
        ttk.Label(title_frame, text="4D-STEM Processor",
                  font=('Arial', 18, 'bold')).pack()
        ttk.Label(title_frame,
                  text="DPC/iDPC + SSB + Strain + Orientation + Ptychography",
                  font=('Arial', 11)).pack()

        input_frame = ttk.LabelFrame(self.root, text="Input", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(input_frame, text="DM4 Folder:").grid(row=0, column=0,
                                                        sticky=tk.W)
        self.input_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_var,
                  width=60).grid(row=0, column=1, padx=5)
        ttk.Button(input_frame, text="Browse...",
                   command=self._browse_input).grid(row=0, column=2)

        output_frame = ttk.LabelFrame(self.root, text="Output", padding=10)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(output_frame, text="Output Folder:").grid(row=0, column=0,
                                                            sticky=tk.W)
        self.output_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_var,
                  width=60).grid(row=0, column=1, padx=5)
        ttk.Button(output_frame, text="Browse...",
                   command=self._browse_output).grid(row=0, column=2)

        opt_frame = ttk.LabelFrame(self.root, text="Options", padding=10)
        opt_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(opt_frame, text="Scan crop size:").grid(row=0, column=0,
                                                          sticky=tk.W)
        self.crop_var = tk.IntVar(value=128)
        ttk.Spinbox(opt_frame, from_=32, to=512, increment=32,
                    textvariable=self.crop_var,
                    width=10).grid(row=0, column=1, sticky=tk.W)

        analysis_frame = ttk.LabelFrame(self.root, text="Analysis Modules",
                                        padding=10)
        analysis_frame.pack(fill=tk.X, padx=10, pady=5)
        self.opt_dpc = tk.BooleanVar(value=True)
        self.opt_ssb = tk.BooleanVar(value=True)
        # Heavy per-pixel crystallography modules default OFF: on a 128x128
        # scan they dominate total runtime by an order of magnitude.
        self.opt_strain = tk.BooleanVar(value=False)
        self.opt_orient = tk.BooleanVar(value=False)
        self.opt_peaks = tk.BooleanVar(value=False)
        self.opt_ptycho = tk.BooleanVar(value=False)

        ttk.Checkbutton(analysis_frame, text="DPC/iDPC Phase",
                        variable=self.opt_dpc).grid(row=0, column=0,
                                                    sticky=tk.W)
        ttk.Checkbutton(analysis_frame, text="SSB Reconstruction",
                        variable=self.opt_ssb).grid(row=0, column=1,
                                                    sticky=tk.W)
        ttk.Checkbutton(analysis_frame, text="Strain Mapping",
                        variable=self.opt_strain).grid(row=0, column=2,
                                                       sticky=tk.W)
        ttk.Checkbutton(analysis_frame, text="Orientation Mapping",
                        variable=self.opt_orient).grid(row=1, column=0,
                                                       sticky=tk.W)
        ttk.Checkbutton(analysis_frame, text="Peak Pairs",
                        variable=self.opt_peaks).grid(row=1, column=1,
                                                      sticky=tk.W)
        ttk.Checkbutton(analysis_frame, text="Ptychography (ePIE)",
                        variable=self.opt_ptycho).grid(row=1, column=2,
                                                       sticky=tk.W)
        ttk.Label(analysis_frame,
                  text="BF/ADF/ABF virtual images and E-field maps are always "
                       "generated").grid(row=2, column=0, columnspan=3,
                                         sticky=tk.W, pady=(5, 0))

        ptycho_frame = ttk.Frame(analysis_frame)
        ptycho_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=5)
        ttk.Label(ptycho_frame, text="ePIE iterations:").pack(side=tk.LEFT)
        self.epie_iter_var = tk.IntVar(value=20)
        ttk.Spinbox(ptycho_frame, from_=5, to=100, increment=5,
                    textvariable=self.epie_iter_var,
                    width=6).pack(side=tk.LEFT, padx=5)

        # 取向指标化用的晶体结构（决定模板库）
        struct_frame = ttk.Frame(analysis_frame)
        struct_frame.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        ttk.Label(struct_frame, text="Structure (orientation):").pack(side=tk.LEFT)
        self.structure_var = tk.StringVar(value=next(iter(STRUCTURE_PRESETS)))
        struct_box = ttk.Combobox(struct_frame, width=16, state='readonly',
                                  textvariable=self.structure_var,
                                  values=list(STRUCTURE_PRESETS))
        struct_box.pack(side=tk.LEFT, padx=5)

        # 高级数值参数（有科学含义，默认值与 core 保持一致）
        adv_frame = ttk.Frame(analysis_frame)
        adv_frame.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        ttk.Label(adv_frame, text="iDPC reg.:").pack(side=tk.LEFT)
        self.idpc_reg_var = tk.DoubleVar(value=1e-3)
        ttk.Spinbox(adv_frame, from_=0.0, to=0.1, increment=0.0005,
                    textvariable=self.idpc_reg_var, width=8,
                    format='%0.4f').pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(adv_frame, text="SSB defocus (rad):").pack(side=tk.LEFT)
        self.ssb_defocus_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(adv_frame, from_=-10.0, to=10.0, increment=0.5,
                    textvariable=self.ssb_defocus_var, width=8,
                    format='%0.1f').pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(adv_frame, text="ePIE scan step (px):").pack(side=tk.LEFT)
        self.scan_step_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(adv_frame, from_=0.25, to=2.0, increment=0.25,
                    textvariable=self.scan_step_var, width=8,
                    format='%0.2f').pack(side=tk.LEFT, padx=2)

        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(fill=tk.X)
        self.start_btn = ttk.Button(btn_frame, text="Start Processing",
                                    command=self._start_processing)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Stop",
                                   command=self._stop_processing,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        prog_frame = ttk.Frame(self.root, padding=(10, 0))
        prog_frame.pack(fill=tk.X)
        self.progress = ttk.Progressbar(prog_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(prog_frame, textvariable=self.status_var).pack()

        log_frame = ttk.LabelFrame(self.root, text="Processing Log",
                                   padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = tk.Text(log_frame, height=15, font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                  command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _browse_input(self):
        path = filedialog.askdirectory(title="Select DM4 Data Folder")
        if path:
            self.input_var.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_var.set(path)

    # ---- thread-safe messaging ----
    def _emit(self, kind, payload):
        self.msg_queue.put((kind, payload))

    def _log(self, msg):
        self._emit('log', msg)

    def _poll_messages(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == 'log':
                    self.log_text.insert(tk.END, payload + "\n")
                    self.log_text.see(tk.END)
                elif kind == 'status':
                    self.status_var.set(payload)
                elif kind == 'progress':
                    self.progress['value'] = payload
                elif kind == 'done':
                    self._on_done(payload)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_messages)

    def _on_done(self, summary):
        self.processing = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        messagebox.showinfo("Complete", summary)

    # ---- control ----
    def _start_processing(self):
        input_dir = self.input_var.get().strip()
        output_dir = self.output_var.get().strip()
        if not input_dir:
            messagebox.showerror("Error", "Please select input folder!")
            return
        if not os.path.isdir(input_dir):
            messagebox.showerror(
                "Error", f"Input folder does not exist:\n{input_dir}")
            return
        if not output_dir:
            messagebox.showerror("Error", "Please select output folder!")
            return
        if not dm4_io.ncempy_available():
            messagebox.showerror(
                "Error",
                "ncempy is not available, so DM4 files cannot be read.\n"
                "Install it with: pip install ncempy")
            return

        # Fail fast on an unusable output location instead of after the
        # first dataset has already been processed.
        try:
            os.makedirs(output_dir, exist_ok=True)
            probe = os.path.join(output_dir, '.write_probe')
            with open(probe, 'w') as fp:
                fp.write('ok')
            os.remove(probe)
        except OSError as exc:
            messagebox.showerror(
                "Error",
                f"Output folder is not writable:\n{output_dir}\n\n{exc}")
            return

        # Snapshot every option here in the MAIN thread. Tkinter variables
        # are not thread-safe, so the worker thread must only ever see
        # plain Python values.
        try:
            scan_crop = int(self.crop_var.get())
            epie_iterations = int(self.epie_iter_var.get())
            dpc_regularization = float(self.idpc_reg_var.get())
            ssb_defocus = float(self.ssb_defocus_var.get())
            epie_scan_step = float(self.scan_step_var.get())
        except (tk.TclError, ValueError, TypeError):
            messagebox.showerror(
                "Error",
                "Scan crop size, ePIE iterations, iDPC regularization,\n"
                "SSB defocus and ePIE scan step must be numbers.")
            return
        if not 8 <= scan_crop <= 4096:
            messagebox.showerror(
                "Error", "Scan crop size must be between 8 and 4096.")
            return
        if not 1 <= epie_iterations <= 1000:
            messagebox.showerror(
                "Error", "ePIE iterations must be between 1 and 1000.")
            return
        if not 0.0 <= dpc_regularization <= 0.1:
            messagebox.showerror(
                "Error", "iDPC regularization must be between 0 and 0.1.")
            return
        if not -10.0 <= ssb_defocus <= 10.0:
            messagebox.showerror(
                "Error", "SSB defocus must be between -10 and 10 rad.")
            return
        if not 0.25 <= epie_scan_step <= 2.0:
            messagebox.showerror(
                "Error", "ePIE scan step must be between 0.25 and 2.0 px.")
            return

        options = {
            'scan_crop': scan_crop,
            'dpc': bool(self.opt_dpc.get()),
            'ssb': bool(self.opt_ssb.get()),
            'strain': bool(self.opt_strain.get()),
            'orient': bool(self.opt_orient.get()),
            'peaks': bool(self.opt_peaks.get()),
            'ptycho': bool(self.opt_ptycho.get()),
            'epie_iterations': epie_iterations,
            'epie_scan_step': epie_scan_step,
            'dpc_regularization': dpc_regularization,
            'ssb_defocus': ssb_defocus,
            'structure': self.structure_var.get(),
        }

        if os.path.abspath(input_dir) == os.path.abspath(output_dir):
            if not messagebox.askyesno(
                    "Confirm",
                    "Input and output folders are the same.\n"
                    "Result subfolders will be written into the data "
                    "folder. Continue?"):
                return

        dm4_files = []
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith('.dm4'):
                    dm4_files.append(os.path.join(root, f))
        dm4_files.sort()

        if not dm4_files:
            messagebox.showwarning("Warning",
                                   f"No DM4 files found in:\n{input_dir}")
            return

        self._log(f"Found {len(dm4_files)} DM4 files")
        for f in dm4_files:
            self._log(f"  {os.path.basename(f)} "
                      f"({os.path.getsize(f) / 1024 ** 3:.2f} GB)")

        self.processing = True
        self.results = []
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress['maximum'] = len(dm4_files)
        self.progress['value'] = 0

        thread = threading.Thread(target=self._process_files,
                                  args=(dm4_files, output_dir, options),
                                  daemon=True)
        thread.start()

    def _stop_processing(self):
        self.processing = False
        self._log("\nProcessing stopped by user (finishing current step)...")

    def _should_stop(self):
        return not self.processing

    def _process_files(self, dm4_files, output_base, options):
        total = len(dm4_files)
        opts = {'dpc': options['dpc'], 'ssb': options['ssb'],
                'dpc_regularization': options['dpc_regularization'],
                'ssb_defocus': options['ssb_defocus']}
        used_names = {}
        run_records = []

        for i, dm4_path in enumerate(dm4_files):
            if not self.processing:
                break
            basename = os.path.splitext(os.path.basename(dm4_path))[0]
            name = (basename.replace(' ', '_')
                    .replace('(', '').replace(')', ''))
            # DM4 files with the same basename in different subfolders must
            # not silently overwrite each other's output directory.
            if name in used_names:
                used_names[name] += 1
                name = f"{name}_{used_names[name]}"
                self._log(f"  Name collision -> output subdir '{name}'")
            else:
                used_names[name] = 1
            output_dir = os.path.join(output_base, name)

            self._emit('status', f"Processing {i + 1}/{total}: {basename}")
            self._log(f"\n{'=' * 60}")
            self._log(f"[{i + 1}/{total}] {basename}")
            self._log(f"{'=' * 60}")

            record = {'name': name, 'file': os.path.basename(dm4_path),
                      'status': 'failed', 'error': ''}
            try:
                workspace = process_dm4_file(
                    dm4_path, output_dir,
                    scan_crop=options['scan_crop'],
                    opts=opts,
                    log=self._log,
                    should_stop=self._should_stop,
                    name=name,
                )
                if workspace is None:
                    if self.processing:
                        self._log("  ERROR: processing returned no result")
                    break
                self.results.append(workspace['meta'])
                record['status'] = 'ok'
                record['elapsed_sec'] = workspace['meta']['elapsed_sec']
                self._run_additional_analyses(workspace, options)
            except Exception as exc:
                self._log(f"  ERROR: {exc}")
                self._log(traceback.format_exc())
                record['error'] = str(exc)
            finally:
                run_records.append(record)
                self._emit('progress', i + 1)

        self._log(f"\n{'=' * 60}")
        if self.processing:
            self._log(f"PROCESSING COMPLETE - {len(self.results)}/{total}")
        else:
            self._log(f"PROCESSING STOPPED - {len(self.results)}/{total}")
        self._log(f"{'=' * 60}")
        self._log(f"{'Name':<25} {'Det':<10} {'Alpha':<8} {'Corr':<10}")
        self._log(f"{'-' * 53}")
        for r in self.results:
            det = f"{r['det_shape'][0]}x{r['det_shape'][1]}"
            corr = (f"{r['dpc_ssb_correlation']:.4f}"
                    if r['dpc_ssb_correlation'] is not None else "n/a")
            self._log(f"{r['name']:<25} {det:<10} {r['alpha']:<8} {corr:<10}")

        # Persist the run summary (the log window is lost on close). Every
        # file gets a row, including failed ones, so the CSV is a complete
        # audit of the run rather than only the successes.
        csv_path = os.path.join(output_base, 'summary.csv')
        try:
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write("name,file,status,elapsed_sec,scan_shape,det_shape,"
                        "alpha,dpc_ssb_correlation,scan_step_nm,error\n")
                meta_by_name = {}
                for meta in self.results:
                    meta_by_name[meta['name']] = meta
                for rec in run_records:
                    meta = meta_by_name.get(rec['name'])
                    if meta is not None:
                        corr_csv = (f"{meta['dpc_ssb_correlation']:.6f}"
                                    if meta['dpc_ssb_correlation'] is not None
                                    else "")
                        f.write(f"{rec['name']},{rec['file']},"
                                f"{rec['status']},"
                                f"{rec.get('elapsed_sec', '')},"
                                f"{meta['scan_shape'][0]}x{meta['scan_shape'][1]},"
                                f"{meta['det_shape'][0]}x{meta['det_shape'][1]},"
                                f"{meta['alpha']},{corr_csv},"
                                f"{meta['scan_step_nm']},"
                                f"\"{rec['error']}\"\n")
                    else:
                        f.write(f"{rec['name']},{rec['file']},"
                                f"{rec['status']},,,,,,,"
                                f"\"{rec['error']}\"\n")
            self._log(f"Summary written: {csv_path}")
        except OSError as exc:
            self._log(f"  Could not write summary.csv: {exc}")

        self._emit('status', f"Done! {len(self.results)}/{total} datasets "
                             f"processed.")
        self._emit('done', f"Processing complete!\n"
                           f"{len(self.results)}/{total} datasets processed.\n"
                           f"Results saved to:\n{output_base}")

    def _run_additional_analyses(self, workspace, options):
        data = workspace['data']
        center = workspace['center']
        alpha = workspace['alpha']
        com = workspace['com']
        name = workspace['name']
        output_dir = workspace['output_dir']
        scale_nm = workspace['scale_nm']
        stop = self._should_stop

        if options['strain']:
            self._log("  Running Strain Mapping (Bragg disk tracking)...")
            self._log("    (reference = whole-image median -> relative "
                      "strain)")
            try:
                strain_result = run_strain_mapping(
                    data, center, alpha, method='bragg',
                    name=name, output_dir=output_dir,
                    should_stop=stop, verbose=False, progress=self._log)
                if strain_result.get('cancelled'):
                    self._log("    Strain mapping cancelled")
                    return
                strain = strain_result['strain']
                n_ok = int(np.isfinite(strain['eps_xx']).sum())
                self._log(f"    indexed {n_ok}/{strain['eps_xx'].size} pixels, "
                          f"{len(strain_result['reference_spots'])} ref spots")
                self._log(f"    eps_xx=[{np.nanmin(strain['eps_xx']):.4f}, "
                          f"{np.nanmax(strain['eps_xx']):.4f}]")
            except Exception as exc:
                self._log(f"    Strain ERROR: {exc}")
                self._log(traceback.format_exc())

        if options['orient'] and not stop():
            self._log("  Running Orientation Mapping (template matching)...")
            try:
                structure = options['structure']
                orient_result = run_orientation_mapping(
                    data, center, alpha, method='template',
                    structure=structure,
                    name=name, output_dir=output_dir,
                    should_stop=stop, verbose=False, progress=self._log)
                if orient_result['orientation'].get('cancelled'):
                    self._log("    Orientation mapping cancelled")
                    return
                orient = orient_result['orientation']
                zi = orient['zone_index']
                found = sorted(set(zi.ravel().tolist()) - {-1})
                labels = orient['zone_labels']
                self._log(f"    structure: {structure}, "
                          f"indexed {(zi >= 0).sum()}/{zi.size} pixels")
                self._log("    zones: " + ", ".join(
                    f"[{labels[t][0]}{labels[t][1]}{labels[t][2]}]"
                    f"({(zi == t).sum()})" for t in found[:8]))
            except Exception as exc:
                self._log(f"    Orientation ERROR: {exc}")
                self._log(traceback.format_exc())

        if options['peaks'] and not stop():
            self._log("  Running Peak Pairs...")
            try:
                pp_result = peak_pairs_mapping(data, center, alpha,
                                               should_stop=stop)
                if pp_result and pp_result.get('cancelled'):
                    self._log("    Peak Pairs cancelled")
                    return
                if pp_result:
                    plot_peak_pairs_results(pp_result, name, output_dir)
                    self._log(f"    Pairs found: {pp_result['n_pairs']}")
                else:
                    self._log("    Peak Pairs: no peaks found, skipped")
            except Exception as exc:
                self._log(f"    Peak Pairs ERROR: {exc}")
                self._log(traceback.format_exc())

        if options['ptycho'] and not stop():
            self._log("  Running Ptychography (ePIE)...")
            try:
                ptycho_result = run_ptychography(
                    data, center, alpha, method='epie',
                    n_iterations=options['epie_iterations'],
                    step_size=0.5,
                    scan_step=options.get('epie_scan_step', 1.0),
                    name=name,
                    output_dir=output_dir, verbose=False,
                    should_stop=stop)
                if ptycho_result.get('cancelled'):
                    self._log("    ePIE cancelled")
                    return
                self._log(f"    ePIE phase: "
                          f"[{ptycho_result['phase'].min():.3f}, "
                          f"{ptycho_result['phase'].max():.3f}]")
                if ptycho_result['errors']:
                    self._log(f"    Final error: "
                              f"{ptycho_result['errors'][-1]:.6f}")
            except Exception as exc:
                self._log(f"    Ptychography ERROR: {exc}")
                self._log(traceback.format_exc())


def main():
    root = tk.Tk()
    app = STEMProcessorApp(root)

    # The packaged exe runs windowed (no console): an unhandled exception
    # would otherwise die silently. Route main-thread and Tk-callback
    # exceptions to an error dialog.
    def _report_unhandled(exc_type, exc_value, exc_tb):
        text = "".join(traceback.format_exception(exc_type, exc_value,
                                                  exc_tb))
        try:
            messagebox.showerror("Unexpected error", text)
        except tk.TclError:
            pass

    sys.excepthook = _report_unhandled
    root.report_callback_exception = _report_unhandled
    root.mainloop()


if __name__ == '__main__':
    main()
