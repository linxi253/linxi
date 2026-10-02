"""Opt-in full-stack validation on real TIFF files; inputs are never modified."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import tifffile

from image_filters import process_image
from tif_io import TifDocument, list_tif_files, safe_output_path
from version import __version__

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for item in args.inputs:
        sources.extend(map(Path, list_tif_files(item)) if item.is_dir() else [item])
    sources = list(dict.fromkeys(path.resolve() for path in sources))
    if not sources:
        parser.error('No TIFF files found in the input folders')
    params = {'exposure': 20, 'contrast': 10}
    report = {'version': __version__, 'filters': params, 'results': []}
    for source in sources:
        start = time.monotonic()
        doc = TifDocument(source)
        print(f'Processing {source.name}: {doc.num_frames} frames, {doc.shape}', flush=True)
        original_hash = sha256(source)
        output = safe_output_path(args.output_dir, source)

        def progress(current, total):
            if current % 100 == 0 or current == total:
                print(f'  written {current}/{total}', flush=True)

        doc.save_processed(output, params, progress)
        changed_frames = 0
        with tifffile.TiffFile(output) as result:
            assert result.byteorder == doc.byteorder
            assert result.series[0].shape == doc.shape
            assert result.series[0].axes == doc.axes
            assert result.series[0].dtype == doc.dtype
            assert len(result.series[0].pages) == doc.num_frames
            for index, raw in enumerate(doc._iter_raw_frames()):
                work, alpha = doc._to_processing_frame(raw)
                expected = doc._from_processing_frame(process_image(work, params), alpha)
                actual = result.series[0].pages[index].asarray()
                np.testing.assert_array_equal(actual, expected)
                changed_frames += int(not np.array_equal(actual, raw))
            assert (result.pages[0].description or None) == doc.description
            with tifffile.TiffFile(source) as original:
                if original.is_imagej:
                    for key in ('Info', 'Labels', 'Properties'):
                        assert result.imagej_metadata.get(key) == original.imagej_metadata.get(key)
        assert sha256(source) == original_hash, 'Source changed during validation'
        manifest = json.loads(Path(f'{output}.filter.json').read_text(encoding='utf-8'))
        assert manifest['tiff']['frames'] == doc.num_frames
        assert manifest['application']['version'] == __version__
        report['results'].append({
            'source': str(source), 'source_sha256': original_hash, 'output': output,
            'shape': doc.shape, 'axes': doc.axes, 'dtype': str(doc.dtype),
            'frames_verified': doc.num_frames, 'changed_frames': changed_frames,
            'source_unchanged': True, 'passed': True,
            'elapsed_seconds': round(time.monotonic() - start, 2),
        })
        report_path = args.output_dir / 'validation.json'
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'  PASS: verified every pixel in {doc.num_frames} frames; source unchanged', flush=True)
    print(f'Validation report: {report_path.resolve()}', flush=True)


if __name__ == '__main__':
    main()
