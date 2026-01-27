from __future__ import annotations

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_train_and_evaluate_writes_artifact(tmp_path, monkeypatch):
    from data_reuploading.lib import runner as dr_runner

    cfg = load_runtime_ready_config()
    markers: dict[str, bool] = {}

    def fake_reproduce(cfg_arg, run_dir):
        markers["figure_5"] = True
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "figure_5_results.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(dr_runner, "reproduce_figure_5", fake_reproduce)

    dr_runner.train_and_evaluate(cfg, tmp_path)

    assert markers.get("figure_5")
    assert (tmp_path / "figure_5_results.json").exists()
