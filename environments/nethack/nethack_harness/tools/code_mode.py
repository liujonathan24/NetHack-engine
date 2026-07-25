"""
nethack_harness.tools.code_mode
======================

Glyphbox-style code execution as a single tool. Lets the agent issue Python
loops directly against a curated `nh` namespace instead of multi-call skill
sequences.

v0 capabilities:

  * `nh.move(direction)` / `nh.attack(direction)` / `nh.descend()` etc.
  * `nh.autoexplore(max_steps)` — same as the skill but in a code namespace
  * `nh.add_note(key, text)` / `nh.recall(query)`
  * `nh.wiki_lookup(entity)` / `nh.wiki_search(query)`
  * `nh.status` / `nh.inventory` / `nh.map_view` — read-only view of state
  * `Direction` / `Position` constants

Safety:

  * AST validator that blocks `import os/sys/subprocess`, `exec/eval/open`,
    dunder attribute access (`__class__`, `__dict__`, etc.).
  * Hard runtime cap via `signal.SIGALRM` (Unix). 5s default.

This is Track B prep. v0 wires the executor and the safety; we leave the
RLM-native sub-tools (`summarize`, `plan`, `recall_lm`) for week 2 when the
prime-rl inference server is available.

References:
  - glyphbox source: github.com/kenforthewin/glyphbox
  - Recursive Language Models (Zhang/Kraska/Khattab, arXiv 2512.24601)
"""

from __future__ import annotations

import ast
import signal
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .skills import SkillResult


# Names we refuse to allow inside the executor namespace.
_FORBIDDEN_NAMES = frozenset({
    "exec", "eval", "compile", "open", "input", "__import__",
    "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "hasattr", "type", "object", "super",
})
# Modules we refuse to import (we permit no imports at all by default).
_FORBIDDEN_IMPORTS = frozenset({
    "os", "sys", "subprocess", "shutil", "socket", "ctypes",
    "importlib", "builtins", "pickle", "marshal", "fcntl",
})


class CodeModeError(Exception):
    """Raised when user code violates the safety policy or times out."""


# ---------- AST validator ----------

class _SafetyValidator(ast.NodeVisitor):
    """Walk an AST and raise CodeModeError on anything unsafe."""

    def visit_Import(self, node: ast.Import) -> None:
        # We allow no imports at all — the namespace already has what's needed.
        names = [alias.name for alias in node.names]
        raise CodeModeError(f"Imports are not allowed in code mode (got {names!r}).")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise CodeModeError(f"`from {node.module} import ...` is not allowed.")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if attr.startswith("__") and attr.endswith("__"):
            raise CodeModeError(f"Dunder attribute access ({attr!r}) is not allowed.")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_NAMES:
            raise CodeModeError(f"Use of {node.id!r} is not allowed.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Already covered by visit_Name for direct calls; this catches things
        # like `(eval)("...")`. Most pragmatic check: just generic_visit.
        self.generic_visit(node)


def validate_source(source: str) -> None:
    """Raise CodeModeError if `source` violates the safety policy."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as e:
        raise CodeModeError(f"Syntax error: {e}") from e
    _SafetyValidator().visit(tree)


# ---------- runtime ----------

@dataclass
class CodeModeResult:
    stdout: str
    error: Optional[str] = None
    actions_taken: list[int] = field(default_factory=list)


class MapView:
    """Read-only structural view of the map for code-mode agents."""
    def __init__(self, model):
        self._m = model

    @property
    def player(self):
        return self._m.player

    @property
    def entities(self):
        return list(self._m.entities)

    def at(self, x, y):
        for e in self._m.entities:
            if e.x == x and e.y == y:
                return e
        return None

    def _of(self, kind):
        return [e for e in self._m.entities if e.kind == kind]

    @property
    def monsters(self):
        return self._of("monster")

    @property
    def stairs(self):
        return self._of("stair")


class _NhNamespace:
    """The `nh` object the executor sees. Wraps an env + skill registry."""

    def __init__(self, env, structured_obs, journal=None, action_log: Optional[list[int]] = None,
                 sub_lm: Optional["SubLM"] = None, raw_obs=None):
        self._env = env
        self._obs = structured_obs
        self._raw_obs = raw_obs
        self._journal = journal
        self._log = action_log if action_log is not None else []
        self._sub_lm = sub_lm or _default_sub_lm()

    # ----- read-only views -----

    @property
    def status(self) -> dict:
        return dict(self._obs.status) if self._obs is not None else {}

    @property
    def inventory(self) -> list:
        return list(self._obs.inventory) if self._obs is not None else []

    @property
    def map_view(self) -> str:
        return self._obs.map_view if self._obs is not None else ""

    @property
    def map(self) -> Optional["MapView"]:
        """Read-only structured map (entities w/ coords, player, terrain).

        Built lazily from the raw NLE obs threaded through run_user_code.
        Returns None when no raw obs is available."""
        if self._raw_obs is None:
            return None
        from nethack_core import build_map_model
        return MapView(build_map_model(self._raw_obs))

    @property
    def character(self) -> dict:
        return dict(self._obs.character) if self._obs is not None else {}

    @property
    def under_player(self):
        """Tile under the @ (e.g. 'stairs DOWN (>) — call descend...').
        Critically tells the code-mode user when they're standing on stairs
        because @ in the map hides it. Same source as the skill-mode
        `=== UNDER PLAYER ===` block."""
        return getattr(self._obs, "under_player", None) if self._obs is not None else None

    @property
    def adjacent(self) -> dict:
        """8-neighborhood of the player; stair glyphs auto-labeled."""
        return dict(getattr(self._obs, "adjacent", {}) or {}) if self._obs is not None else {}

    # ----- action methods. These dispatch the corresponding skill from the
    # registry, append the resulting NLE action sequence to `_log`, and return
    # immediately. The verifiers env_response that called run_user_code is
    # responsible for applying `_log` to env.step() after the user code
    # returns. This batch-then-flush model avoids stepping the env mid-code-
    # execution (which would change observations and make the user code's
    # reasoning go stale).

    def _dispatch(self, skill: str, **kwargs) -> "SkillResult":
        from .skills import registry
        result = registry.call(skill, self._env, self._obs, **kwargs)
        if result.actions:
            self._log.extend(int(a) for a in result.actions)
        return result

    def move(self, direction: str) -> None:
        self._dispatch("move", direction=direction)

    def attack(self, direction: str) -> None:
        self._dispatch("attack", direction=direction)

    def descend(self) -> None:
        self._dispatch("descend")

    def search(self) -> None:
        self._dispatch("search")

    def pickup(self) -> None:
        self._dispatch("pickup")

    def autoexplore(self, max_steps: int = 30) -> None:
        self._dispatch("autoexplore", max_steps=max_steps)

    def move_to(self, x: int, y: int) -> None:
        self._dispatch("move_to", x=x, y=y)

    def add_note(self, key: str, text: str) -> None:
        if self._journal is not None:
            self._journal.add_note(key, text)

    def recall(self, query: str) -> list:
        if self._journal is None:
            return []
        return self._journal.recall(query)

    def wiki_lookup(self, entity: str):
        from .wiki import get_index
        return get_index().lookup(entity)

    def wiki_search(self, query: str, k: int = 3) -> list:
        from .wiki import get_index
        return get_index().search(query, k=k)

    # ----- sub-LM tools (Track B / RLM core) -----
    #
    # These route through a `SubLM` backend. The default backend is offline
    # and returns deterministic stubs (so tests pass with no API access). To
    # plug in a real inference server, build a SubLM subclass and pass it via
    # `run_user_code(..., sub_lm=YourSubLM())`. The verifiers env builds the
    # SubLM once per rollout so the same client is reused.

    def summarize(self, slice_text: str, query: Optional[str] = None) -> str:
        return self._sub_lm.summarize(slice_text, query=query)

    def plan(self, objective: str, horizon: int = 5) -> list[str]:
        return self._sub_lm.plan(objective, horizon=horizon)

    def recall_lm(self, query: str) -> str:
        # Combines journal recall + obs context via the sub-LM.
        notes = self._journal.recall(query) if self._journal is not None else []
        ctx = "\n".join(f"- {n}" for n in notes) or "(no notes)"
        return self._sub_lm.recall(query, context=ctx)


def run_user_code(
    source: str,
    env,
    structured_obs,
    journal=None,
    timeout_seconds: int = 5,
    raw_obs=None,
) -> CodeModeResult:
    """
    Validate and execute `source` against a controlled namespace.

    Returns a CodeModeResult capturing stdout, the error message (if any),
    and the list of NLE actions the code asked to take.
    """
    try:
        validate_source(source)
    except CodeModeError as e:
        return CodeModeResult(stdout="", error=str(e))

    nh = _NhNamespace(env, structured_obs, journal, raw_obs=raw_obs)
    import io
    import contextlib

    namespace = {
        "nh": nh,
        "Direction": _DIRECTIONS,
        "Position": _Position,
        "__builtins__": _safe_builtins(),
    }

    buf = io.StringIO()
    error: Optional[str] = None

    def _alarm_handler(_signum, _frame):
        raise CodeModeError(f"Code timed out after {timeout_seconds}s.")

    # SIGALRM is Unix-only; on platforms without it we skip the cap and rely
    # on validate_source to keep things tractable.
    have_alarm = hasattr(signal, "SIGALRM")
    if have_alarm:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout_seconds)

    try:
        with contextlib.redirect_stdout(buf):
            exec(source, namespace)
    except CodeModeError as e:
        error = str(e)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        if have_alarm:
            signal.alarm(0)

    return CodeModeResult(
        stdout=buf.getvalue(),
        error=error,
        actions_taken=list(nh._log),
    )


def _safe_builtins() -> dict:
    """A minimal __builtins__ dict. No file I/O, no introspection."""
    import builtins
    allowed = {
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "int", "isinstance", "issubclass", "iter", "len", "list", "map",
        "max", "min", "next", "print", "range", "reversed", "round", "set",
        "slice", "sorted", "str", "sum", "tuple", "zip",
        "True", "False", "None",
    }
    return {name: getattr(builtins, name) for name in allowed if hasattr(builtins, name)}


# ---------- helpers exposed to user code ----------

_DIRECTIONS = type("Direction", (), {
    "N": "N", "NE": "NE", "E": "E", "SE": "SE",
    "S": "S", "SW": "SW", "W": "W", "NW": "NW",
    "WAIT": ".",
})


@dataclass(frozen=True)
class _Position:
    x: int
    y: int


# ---------- Sub-LM backend ----------


class SubLM:
    """Backend for nh.summarize/plan/recall_lm. Replace with a real LM client
    (e.g. prime-rl inference server) by subclassing and overriding the methods.

    The default OfflineSubLM returns deterministic stubs. Useful for tests
    and for demonstrating the API shape without burning tokens.
    """

    def summarize(self, text: str, query: Optional[str] = None) -> str:
        raise NotImplementedError

    def plan(self, objective: str, horizon: int = 5) -> list[str]:
        raise NotImplementedError

    def recall(self, query: str, context: str = "") -> str:
        raise NotImplementedError


class OfflineSubLM(SubLM):
    """Deterministic stub. summarize/plan/recall return short, structured
    placeholders that include their inputs so callers can verify wiring."""

    def summarize(self, text: str, query: Optional[str] = None) -> str:
        head = text.strip().splitlines()[0][:120] if text.strip() else ""
        if query:
            return f"[offline-summary] (query={query!r}) {head}…"
        return f"[offline-summary] {head}…"

    def plan(self, objective: str, horizon: int = 5) -> list[str]:
        return [f"[offline-plan step {i+1}/{horizon}] toward: {objective}" for i in range(horizon)]

    def recall(self, query: str, context: str = "") -> str:
        first_note = context.splitlines()[0] if context.strip() else "(no notes)"
        return f"[offline-recall] q={query!r} → {first_note}"


def _default_sub_lm() -> SubLM:
    return OfflineSubLM()


# ---------- real Prime-backed Sub-LM (belief-state distillation) ----------


def _resolve_prime_endpoint(model: str) -> dict:
    """Resolve {base_url, api_key, headers} for `model` from configs/endpoints.toml.

    Mirrors how vf-eval resolves `-m <model>` against the endpoint registry:
    find the endpoint block whose `models` list contains `model` and use its
    url / key-env / headers (e.g. the prime-team block's X-Prime-Team-ID, which
    Prime requires to bill a funded team instead of the $0 personal balance).

    Falls back to the Prime Inference URL + PI_API_KEY when the toml can't be
    found or parsed, so this stays usable outside the repo tree. Env overrides
    SUBLM_BASE_URL / SUBLM_API_KEY / PRIME_TEAM_ID win over everything.
    """
    import os
    import tomllib

    base_url = None
    headers: dict[str, str] = {}
    key_env = "PI_API_KEY"

    # Walk up from cwd looking for configs/endpoints.toml.
    here = os.getcwd()
    toml_path = None
    for _ in range(6):
        cand = os.path.join(here, "configs", "endpoints.toml")
        if os.path.exists(cand):
            toml_path = cand
            break
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    if toml_path is None:
        env_toml = os.getenv("ENDPOINTS_PATH")
        if env_toml and os.path.exists(env_toml):
            toml_path = env_toml

    if toml_path is not None:
        try:
            with open(toml_path, "rb") as f:
                cfg = tomllib.load(f)
            for ep in cfg.get("endpoints", []):
                if model in (ep.get("models") or []):
                    base_url = ep.get("url")
                    key_env = ep.get("key", key_env)
                    headers = dict(ep.get("headers") or {})
                    break
        except Exception:
            pass

    base_url = os.getenv("SUBLM_BASE_URL") or base_url or "https://api.pinference.ai/api/v1"
    api_key = (
        os.getenv("SUBLM_API_KEY")
        or os.getenv(key_env)
        or os.getenv("PI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    team_override = os.getenv("PRIME_TEAM_ID")
    if team_override:
        headers["X-Prime-Team-ID"] = team_override
    return {"base_url": base_url, "api_key": api_key, "headers": headers}


_SUBLM_SYSTEM_PROMPT = (
    "You are a belief-state summarizer for a NetHack-playing agent. Given the "
    "agent's current status line and its running journal notes, write ONE compact "
    "sentence (<=40 words) capturing the agent's situation and what it should do "
    "next: dungeon level, HP posture, notable threats/items, and the immediate "
    "objective. No preamble, no markdown, just the sentence."
)


class PrimeSubLM(SubLM):
    """Real, in-process Sub-LM backed by an OpenAI-compatible chat endpoint.

    Built from a model id (e.g. "google/gemini-3-flash-preview") plus the
    endpoint registry (configs/endpoints.toml) so belief-state distillation
    calls the SAME Prime team-billed path the main policy uses. `summarize`
    returns a real one-sentence LM summary; on any error it falls back to a
    short deterministic head-of-text stub so it never breaks a rollout.

    plan/recall are not used by belief-state; they delegate to a stub so the
    SubLM interface stays complete.
    """

    def __init__(self, model: str, *, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, headers: Optional[dict] = None,
                 timeout_s: float = 30.0, max_tokens: int = 120) -> None:
        import os
        if base_url is None or api_key is None or headers is None:
            resolved = _resolve_prime_endpoint(model)
            base_url = base_url or resolved["base_url"]
            api_key = api_key or resolved["api_key"]
            headers = resolved["headers"] if headers is None else headers
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.headers = dict(headers or {})
        self.timeout_s = float(os.getenv("SUBLM_TIMEOUT_S", timeout_s))
        self.max_tokens = max_tokens
        self._offline = OfflineSubLM()

    def summarize(self, text: str, query: Optional[str] = None) -> str:
        if not self.api_key:
            return self._offline.summarize(text, query)
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=self.headers or None,
            )
            user_msg = text if not query else f"[{query}]\n{text}"
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SUBLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=self.max_tokens,
                timeout=self.timeout_s,
            )
            out = (resp.choices[0].message.content or "").strip()
            return out or self._offline.summarize(text, query)
        except Exception:
            return self._offline.summarize(text, query)

    def plan(self, objective: str, horizon: int = 5) -> list[str]:
        return self._offline.plan(objective, horizon)

    def recall(self, query: str, context: str = "") -> str:
        return self._offline.recall(query, context)
