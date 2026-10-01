from video_extractor import cli
from video_extractor.cli import main
from video_extractor.models import JobState, VideoResult


def test_cli_missing_input_returns_clean_error(capsys) -> None:
    code = main(["--input", "C:/definitely/missing", "--output", "C:/tmp/out"])
    assert code == 1
    captured = capsys.readouterr()
    assert "错误" in captured.err
    assert "输入目录不存在" in captured.err
    assert "FFmpeg" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_missing_sampling_value_returns_clean_error(capsys) -> None:
    code = main(["--input", ".", "--output", ".", "--mode", "target_fps"])
    assert code == 1
    captured = capsys.readouterr()
    assert "错误" in captured.err
    assert "目标帧率或时间间隔" in captured.err


def test_cli_events_go_to_stderr_results_to_stdout(capsys, monkeypatch) -> None:
    """进度事件必须在 stderr，stdout 只保留可 json.load 的最终结果数组。"""

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run_batch(self, input_root, output_root, callback=None):
            if callback:
                callback({"kind": "video_started", "index": 1, "total": 1, "path": "a.mp4"})
                callback({"kind": "warning", "message": "示例告警"})
            return [VideoResult(input_path="a.mp4", state=JobState.COMPLETED, frame_count=3)]

    monkeypatch.setattr(cli, "JobRunner", FakeRunner)
    code = main(["--input", ".", "--output", "."])
    assert code == 0
    captured = capsys.readouterr()
    assert "video_started" in captured.err
    assert "示例告警" in captured.err
    assert "video_started" not in captured.out
    assert '"state": "completed"' in captured.out


def test_cli_returns_130_for_cancelled_batch(capsys, monkeypatch) -> None:
    """取消与失败必须使用不同退出码，脚本才能区分"没做完"和"做错了"。"""

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run_batch(self, input_root, output_root, callback=None):
            return [VideoResult(input_path="a.mp4", state=JobState.CANCELLED)]

    monkeypatch.setattr(cli, "JobRunner", FakeRunner)
    code = main(["--input", ".", "--output", "."])
    assert code == 130


def test_cli_rejects_value_in_all_mode(capsys) -> None:
    """全部帧模式传 --value 不再被静默忽略。"""
    code = main(["--input", ".", "--output", ".", "--mode", "all", "--value", "5"])
    assert code == 1
    captured = capsys.readouterr()
    assert "不接受采样值" in captured.err


def test_cli_accepts_lenient_decode_flag(capsys, monkeypatch) -> None:
    captured_options = {}

    class FakeRunner:
        def __init__(self, options, *args, **kwargs):
            captured_options.update(options.__dict__)

        def run_batch(self, input_root, output_root, callback=None):
            return [VideoResult(input_path="a.mp4", state=JobState.COMPLETED)]

    monkeypatch.setattr(cli, "JobRunner", FakeRunner)
    code = main(["--input", ".", "--output", ".", "--lenient-decode"])
    assert code == 0
    assert captured_options["lenient_decode"] is True
