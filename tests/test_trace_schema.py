"""The versioned NDJSON trace contract shared by the env writer and the viewers."""
import io
import pytest

from nethack_core import trace_schema as ts


def _minimal_record():
    return {
        "turn": 0,
        "raw_grid": ["", ""],
        "status": {"depth": 1, "hitpoints": 14},
        "rendered_user_message": "you see here a rock",
        "tool_calls": [],
        "action_indices": [],
        "reward": 0.0,
    }


def test_to_json_line_stamps_current_version():
    line = ts.to_json_line(_minimal_record())
    assert line.endswith("\n")
    rec = ts.parse_line(line)
    assert rec[ts.SCHEMA_VERSION_KEY] == ts.TRACE_SCHEMA_VERSION
    assert ts.record_version(rec) == ts.TRACE_SCHEMA_VERSION


def test_write_record_roundtrips_through_a_file_object():
    buf = io.StringIO()
    ts.write_record(buf, _minimal_record())
    ts.write_record(buf, {**_minimal_record(), "turn": 1})
    lines = buf.getvalue().splitlines()
    assert len(lines) == 2
    recs = [ts.parse_line(x) for x in lines]
    assert [r["turn"] for r in recs] == [0, 1]


def test_legacy_records_without_stamp_read_as_version_zero():
    assert ts.record_version({"turn": 0}) == 0


def test_parse_line_skips_blank_invalid_and_non_object():
    assert ts.parse_line("") is None
    assert ts.parse_line("   ") is None
    assert ts.parse_line("{not json") is None
    assert ts.parse_line("[1, 2, 3]") is None  # valid JSON, not an object
    assert ts.parse_line('{"turn": 3}') == {"turn": 3}


def test_iter_records_tolerates_malformed_lines(tmp_path):
    p = tmp_path / "trace.ndjson"
    good = ts.to_json_line(_minimal_record())
    p.write_text(good + "\n" + "garbage\n" + "\n" + ts.to_json_line({**_minimal_record(), "turn": 1}))
    recs = ts.read_trace(p)
    assert [r["turn"] for r in recs] == [0, 1]


def test_validate_record_flags_missing_required_and_unknown_fields():
    assert ts.is_valid(_minimal_record())
    problems = ts.validate_record({"turn": 0})
    assert any("missing required field" in p for p in problems)
    problems = ts.validate_record({**_minimal_record(), "bogus": 1})
    assert any("unknown field" in p and "bogus" in p for p in problems)


def test_optional_and_web_console_fields_are_accepted():
    rec = {**_minimal_record(), "checkpoint": "/tmp/x.ckpt", "variant": "web_play",
           "messages": ["hello"], "dlvl": 1, "hp": 14}
    assert ts.is_valid(rec)


# --------------------------------------------------------------------------- #
# schema version 2: reproducibility fields                                     #
# --------------------------------------------------------------------------- #
def test_version_2_fields_are_optional_so_legacy_records_still_validate():
    """Every v2 addition is optional by construction: the pre-versioning traces
    (and the committed CLI-arm reference traces) carry none of them."""
    assert ts.TRACE_SCHEMA_VERSION == 2
    assert ts.is_valid(_minimal_record())
    for field in ("t_mono", "lm_turn", "applied", "actions", "all_messages",
                  "tool_results", "gt_obs"):
        assert field in ts.ALL_FIELDS
        assert field not in ts.REQUIRED_FIELDS


def test_t_mono_is_documented_rather_than_dropped():
    """The writer has emitted it since before versioning and
    `tools/cli_harness_eval/aggregate.py` consumes it, so the schema was the
    stale side of the drift, not the trace."""
    rec = {**_minimal_record(), "t_mono": 812.5}
    assert ts.validate_record(rec) == []


def test_an_unobservable_action_record_is_not_an_empty_one():
    """`[]` used to mean both "executed nothing" and "could not see what was
    executed". Those are different facts and v2 keeps them apart."""
    unknown = ts.empty_action_record("no engine access")
    empty = ts.action_record([])
    assert unknown["recorded"] is False and unknown["reason"]
    assert empty["recorded"] is True and empty["reason"] is None
    assert unknown["n"] == empty["n"] == 0
    assert unknown != empty


def test_a_turn_can_be_recorded_yet_not_replayable():
    """`rollback` rewinds the engine heap directly instead of stepping, so its
    keystrokes are fully observed and still insufficient to reproduce the turn.
    Those are separate flags on purpose."""
    rewound = ts.action_record([27], replayable=False, reason="heap snapshot restore")
    assert rewound["recorded"] is True
    assert rewound["replayable"] is False
    assert rewound["reason"] == "heap snapshot restore"
    assert ts.action_record([27])["replayable"] is True


def test_action_record_keeps_the_bytes_authoritative_and_keys_derived():
    rec = ts.action_record([ord("k"), 27, 13, ord("h"), 4])
    assert rec["bytes"] == [107, 27, 13, 104, 4]
    assert rec["n"] == 5
    assert rec["keys"] == "k<ESC><CR>h^D"
    assert rec["source"] == ts.ACTION_SOURCE_ENGINE_HOOK
    assert rec["keys"] == "".join(ts.key_repr(b) for b in rec["bytes"])


def test_obs_blob_round_trips_every_plane():
    np = pytest.importorskip("numpy")

    class Obs:
        glyphs = np.arange(21 * 79, dtype=np.int16).reshape(21, 79) % 400
        blstats = np.arange(27, dtype=np.int64)
        message = np.zeros(256, dtype=np.uint8)

    blob = ts.encode_obs_blob(Obs(), planes=("glyphs", "blstats", "message"))
    assert blob["codec"] == ts.GT_OBS_CODEC
    out = ts.decode_obs_blob(blob)
    assert set(out) == {"glyphs", "blstats", "message"}
    for name in out:
        ref = getattr(Obs, name)
        assert out[name].dtype == ref.dtype and out[name].shape == ref.shape
        assert (out[name] == ref).all()
    # And it is genuinely compressed, not a JSON integer dump.
    assert len(blob["data"]) < blob["raw_bytes"]


def test_obs_blob_decoding_is_tolerant():
    assert ts.decode_obs_blob(None) == {}
    assert ts.decode_obs_blob({}) == {}
    assert ts.decode_obs_blob({"codec": "future-codec-v9"}) == {}
    assert ts.encode_obs_blob(None) is None
