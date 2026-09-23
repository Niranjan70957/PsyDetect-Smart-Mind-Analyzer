import numpy as np
import pytest

from tools.feature_cache import atomic_write, save_features, valid_cache


def test_corrupted_cache_can_be_rebuilt(tmp_path):
    target = tmp_path / "sample.npz"
    target.write_bytes(b"\0" * 100)
    assert not valid_cache(target, "version")
    features = dict(audio=np.ones(1536), face=np.ones(1536), pipeline_id="version",
                    quality={"duration_seconds": 5}, speech_emotion={}, facial_emotion={})
    save_features(target, features)
    assert valid_cache(target, "version")
    assert not valid_cache(target, "other-version")
    features["audio"][0] = np.nan
    save_features(target, features)
    assert not valid_cache(target, "version")


def test_interrupted_write_preserves_previous_cache(tmp_path):
    target = tmp_path / "report.json"
    target.write_bytes(b"previous")

    def interrupted(stream):
        stream.write(b"incomplete")
        raise OSError("interrupted")

    with pytest.raises(OSError, match="interrupted"):
        atomic_write(target, interrupted)
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]
