import unittest

from tiff_handler import _ome_physical_metadata, _safe_imagej_metadata

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



class TiffSecurityTests(unittest.TestCase):
    def test_imagej_invalid_numeric_fields_are_dropped(self):
        metadata = {
            "unit": "nm",
            "finterval": 0.25,
            "min": "not-a-number",
            "max": None,
            "ranges": [0, "bad"],
            "labels": ["frame A"],
        }
        with self.assertLogs("tiff_handler", level="WARNING"):
            result = _safe_imagej_metadata(metadata)
        self.assertEqual(result["unit"], "nm")
        self.assertEqual(result["finterval"], 0.25)
        self.assertEqual(result["labels"], ["frame A"])
        self.assertNotIn("min", result)
        self.assertNotIn("max", result)
        self.assertNotIn("ranges", result)

    def test_imagej_valid_numeric_fields_are_kept(self):
        metadata = {
            "unit": "nm",
            "finterval": 0.25,
            "min": 1,
            "max": 100,
            "ranges": [1.0, 100.0],
        }
        result = _safe_imagej_metadata(metadata)
        self.assertEqual(result["finterval"], 0.25)
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 100.0)
        self.assertEqual(result["ranges"], [1.0, 100.0])

    def test_ome_external_entity_is_rejected(self):
        xml = """<?xml version="1.0"?>
<!DOCTYPE OME [<!ENTITY secret SYSTEM "file:///C:/Windows/win.ini">]>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels ID="Pixels:0" PhysicalSizeX="&secret;" />
  </Image>
</OME>
"""
        with self.assertLogs("tiff_handler", level="WARNING"):
            result = _ome_physical_metadata(xml)
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
