from __future__ import annotations

from pathlib import Path

import pytest

from sac.config import load_config


def test_load_config_invalid_gamma_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        """
        sac:
          gamma: 1.5
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="gamma"):
        load_config(cfg_path)


def test_load_config_empty_env_id_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad_env.yaml"
    cfg_path.write_text(
        """
        env:
          env_id: ""
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="env_id"):
        load_config(cfg_path)


def test_load_config_unknown_key_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "unknown.yaml"
    cfg_path.write_text(
        """
        sac:
          does_not_exist: 123
        """,
        encoding="utf-8",
    )

    with pytest.raises(TypeError):
        load_config(cfg_path)
