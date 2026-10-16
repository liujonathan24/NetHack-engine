"""Random-play checkpoint/resume fuzz: the 13 fixed cases of tests/_resume_fuzz.py.

Each case plays a random mix of commands in one process, checkpoints at a
random command boundary, then the tail of 300 keys must match field for
field both in a new process resumed from the checkpoint and in-process from
a snapshot taken at the same point.  tools/resume_fuzz.py runs any number.
"""
import pytest

from tests import _resume_fuzz as F

CASES = F.DEFAULT_CASES


def test_default_cases_cover_the_space():
    assert len(CASES) == 13
    assert len({c["role"] for c in CASES}) == 13
    assert {c["depth"] for c in CASES} == {1, 2, 3, 4}
    assert any(c["tune"] for c in CASES)
    wanted = {w for c in CASES for w in c["expect"]}
    assert {"shop", "stairs", "death", "engraving"} <= wanted and wanted & {"temple", "fountain"}


@pytest.mark.parametrize("spec", CASES, ids=[c["id"] for c in CASES])
def test_fuzz_case_resumes_exactly(spec, tmp_path):
    res = F.run_case(spec, tmp_path)
    assert res["status"] != "error", res.get("error")
    assert res["status"] != "died_before_ckpt", res["done_at"]
    assert 60 <= res["ck_step"] <= 300, res["ck_step"]
    for want in spec["expect"]:
        assert res["features"][want], (want, res["features"])
    assert "crash" not in res["inmem"], res["inmem"]["crash"][-400:]
    assert res["inmem"]["first_diff"] is None, res["inmem"]
    assert res["first_diff"] is None, res["first_diff"]
    assert res["steps"] == res["n_after"] and res["n_after"] >= 1, res
