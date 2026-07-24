"""
nethack
=======

The Prime Intellect Environments Hub wrapper for the NetHack training env.

Consumes `nethack_core` and presents it as a verifiers MultiTurnEnv with 
chat-shaped tool calling and a composable rubric.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

import verifiers as vf
from datasets import Dataset

from nethack_core import NetHackCoreEnv
from nethack_harness.memory.journal import Journal
from nethack_core import shape as shape_observation
from nethack_harness.tools.skills import registry as skill_registry, list_skills
from nethack_harness.curriculum.curriculum import get_tier, list_tiers, TierName

try:
    from environments.nethack import harness_overlay as _harness_overlay
except ModuleNotFoundError:
    # Fallback when imported from the package directory (e.g. tests run with
    # cwd=environments/nethack) rather than the repo root, where the
    # `environments.nethack` namespace package resolves.
    import harness_overlay as _harness_overlay


# ---------- verifiers 0.1.14 compat shim ----------
#
# verifiers 0.1.14's v1.utils.sandbox_program_utils.message_from_response
# assumes every tool_call has the OpenAI SDK nested shape (.function.name /
# .function.arguments). Some endpoints (e.g. api.pinference.ai for Qwen) and
# verifiers' own ToolCall dataclass use the flat shape (.name / .arguments).
# Without this shim, vf-eval raises:
#   AttributeError("'ToolCall' object has no attribute 'function'")
# Remove once the upstream PR lands.
def _patch_verifiers_message_from_response() -> None:
    try:
        from verifiers.v1.utils import sandbox_program_utils as _spu
    except ImportError:
        return
    if not hasattr(_spu, "message_from_response"):
        return  # different verifiers build — nothing to patch

    def _safe(response):  # type: ignore[no-redef]
        choice = response.choices[0]
        message = choice.message
        data = {"role": getattr(message, "role", "assistant")}
        content = getattr(message, "content", None)
        if content is not None:
            data["content"] = content
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            packed = []
            for call in tool_calls:
                fn = getattr(call, "function", None)
                name = getattr(fn, "name", None) if fn is not None else getattr(call, "name", None)
                args = getattr(fn, "arguments", None) if fn is not None else getattr(call, "arguments", None)
                packed.append({
                    "id": getattr(call, "id", None),
                    "type": getattr(call, "type", "function"),
                    "function": {"name": name, "arguments": args},
                })
            data["tool_calls"] = packed
        return data

    _spu.message_from_response = _safe


_patch_verifiers_message_from_response()


# ---------- system prompt ----------


# ---------- extracted modules (re-exported for back-compat) ----------
from nethack_harness.prompt.rendering import (
    SYSTEM_PROMPT,
    _strip_blank_rows,
    _glyph_run_encode,
    _inventory_fingerprint,
    _run_length_encode_messages,
    _glyph_to_words,
    _format_obs_balrog,
    _format_obs_glyphbox,
    _format_obs_summarize_reset,
    _descent_status_block,
    _E1_BEARINGS,
    _e1_bearing,
    _e1_classify_frontier,
    _e1_frontiers_block,
    _e1_exploration_block,
    _e1_spatial_belief_block,
    _VARIANT_FORMATTERS,
    _paint_frontiers_on_map,
    format_observation_as_chat,
)
from nethack_harness.helpers import (
    _continual_reset,
    _write_trace_entry,
    _drop_before_last_belief,
    _refinement_directive,
    _ch_build_window,
    _ch_inject_system,
    _ch_save_bootstrap,
    _compact_chat_history,
    _sanitize_assistant_content,
    _STATUS_SIG_RE,
    _compacted_status_signature,
    _dedupe_compacted_runs,
    _msg_role,
    _msg_content,
    _replace_content,
    _one_line_summary,
    _check_halt_condition,
    BELIEF_STATE_INTERVAL,
    _maybe_belief_state_summary,
    _maybe_distill,
    _to_action_indices,
    scout_reward,
    descent_reward,
    success_reward,
    ascension_reward,
    _ASCENSION_MARKERS,
    _DEATH_MARKERS,
    _decode_tty,
    _detect_terminal_outcome,
    _code_tool_adapter,
    _build_skill_adapter_callables,
    _make_run_macro_adapter,
    _make_fixed_direction_adapter,
    _TYPE_MAP,
    _make_skill_adapter,
)
from nethack_harness.prompt.prompt_spec import (
    ObsSpec,
    ToolSpec,
    PromptSpec,
    build_prompt,
    VARIANT_REGISTRY,
    resolve_spec,
    attach_refiner,
)
from nethack_harness.prompt.content import compose_user_content, content_to_text


class NetHackVerifiersEnv(vf.StatefulToolEnv):
    """
    Per-rollout state: a live NetHackCoreEnv plus character + cumulative scout count.

    We subclass StatefulToolEnv because each rollout owns a long-lived NLE
    instance that must be cleanly initialized in setup_state and torn down on
    completion.

    interface: "skill" (default) or "code". In code mode, `env_response` routes
    the model's `code(source=...)` tool call through `code_mode.run_user_code`,
    which executes against an `nh` namespace and produces a list of NLE actions
    that we then step.
    """

    def __init__(
        self,
        *args,
        interface: str = "skill",
        sub_lm=None,
        subgoal_proposer=None,
        # Compaction knobs (survey rec). Set via load_environment kwargs.
        compact_obs: bool = True,
        history_keep_full: int = 5,
        history_drop_after: int = 100,
        belief_state_interval: int = 25,
        journal_render_max_chars: int = 2000,
        # Obs/skill-structure variant for wave-1 experiments. "B1" (default) is
        # the current shipping behavior. "P" is the Continual Harness adaptation:
        # periodic self-refinement turns that prompt the agent to revise its
        # objective and record a lesson note (no NLE step consumed when the
        # agent calls pin_objective/add_note). See docs/PROMPTING_SURVEY.md.
        variant: str = "B1",
        # Detail level for the structured-map variants (JSON/TOON): "full"
        # emits rich entity attrs + RLE grid; "minimal" trims to kind/coord/desc.
        # Threaded onto state["map_detail"] for the per-turn template to read.
        map_detail: str = "full",
        refine_interval: int = 20,
        # Variant R (CPP/GPP summarize-and-reset): when True, get_prompt_messages
        # hard-drops every user/assistant turn that landed before the most-recent
        # belief_state:tN journal note. Combined with belief_state_interval > 0
        # this implements "the belief state IS the memory; chat is disposable."
        summarize_and_reset: bool = False,
        # Per-turn NDJSON trace (raw_grid + rendered_user_message +
        # assistant_message + tool_calls). One file per rollout. Off by default.
        trace_dir: Optional[str] = None,
        # Continual-harness mode: on death, auto-reset NLE and keep playing in
        # the same chat session, preserving journal + belief state across
        # episodes. The rollout terminates when continual_lives is exhausted
        # or the agent ascends.
        continual: bool = False,
        continual_lives: int = 5,
        # Variant CH (Continual Harness, arXiv:2605.09998) full Refiner.
        # `refiner` is a pluggable object satisfying nethack_harness.refiner.Refiner;
        # when None and variant=="CH", we build a TeacherLLMRefiner from
        # refiner_model (or fall back to OfflineRefiner). bootstrap_dir, if set,
        # is used to persist/load the four CH components (prompt addendum,
        # sub-agents, skills, journal) across rollouts.
        refiner: Any = None,
        refiner_model: Optional[str] = None,
        bootstrap_dir: Optional[str] = None,
        # Decouple the teacher Refiner from the obs format: when True, attach the
        # full CH refiner machinery (refiner + sub-agent hooks, system inject,
        # run_macro tool) onto whatever `variant` (obs form) is selected — e.g.
        # variant="JSON", refine=True gives JSON observations PLUS the teacher
        # refiner. variant="CH" implies refine=True (its canonical ASCII obs).
        refine: bool = False,
        # Resolved PromptSpec describing how this rollout builds its prompt
        # (obs form + processors, system prompt, per-turn template, tools). When
        # None we resolve it from `variant` for back-compat; load_environment
        # passes one built post-overlay so the NETHACK_HARNESS seam is honoured.
        spec: Optional[PromptSpec] = None,
        # Game-setup overrides applied to every episode's engine reset (the
        # difficulty/generation knobs from our interface). All None = vanilla
        # NetHack. See load_environment for the shapes.
        setup_tune: Optional[dict] = None,
        setup_modify: Optional[dict] = None,
        setup_level_blob: Optional[str] = None,
        # Pin the character (role-race-alignment-gender, e.g. "Val-hum-neu-fem")
        # for every episode's engine reset. None = engine default (random roll).
        # Threaded to NetHackCoreEnv.reset(character=...). Curriculum tiers own
        # their own character and ignore this.
        setup_character: Optional[str] = None,
        # Names of the skills actually exposed to the model this rollout (the
        # tool schema it was given). Tool calls for anything NOT in this set are
        # rejected in env_response instead of being dispatched against the full
        # registry — otherwise a hallucinated `move(direction=…)` executes even
        # under the netplay set (which withholds it), leaking the low-level
        # primitive into the effective action surface and confounding
        # cross-encoding comparisons. None/empty disables the gate (back-compat).
        allowed_skill_names: Optional[set] = None,
        **kwargs,
    ):
        self.interface = interface
        self._setup_tune = setup_tune
        self._setup_modify = setup_modify
        self._setup_level_blob = setup_level_blob
        self._setup_character = setup_character
        self._allowed_skill_names = set(allowed_skill_names or ())
        # Pluggable LM backends. Both default to None → the rollout-time code
        # falls back to the deterministic Offline* implementations. Swap in
        # prime-rl-backed clients by passing them here from load_environment.
        self.sub_lm = sub_lm
        self.subgoal_proposer = subgoal_proposer
        # Compaction knobs. compact_obs=False reverts to the v0.0.15-era
        # raw rendering (good for replay / debugging / A/B). The history /
        # belief-state / journal knobs let you trade off LM context size
        # against semantic fidelity per run.
        self.compact_obs = compact_obs
        self.history_keep_full = history_keep_full
        self.history_drop_after = history_drop_after
        self.belief_state_interval = belief_state_interval
        self.journal_render_max_chars = journal_render_max_chars
        self.variant = variant
        self.map_detail = map_detail
        # Single predicate gating ALL refiner machinery (teacher construction,
        # separation guard, bootstrap I/O, edit capture, spec hooks). variant=="CH"
        # always implies the refiner; `refine=True` opts ANY other variant in.
        self.refine_enabled = (variant == "CH") or bool(refine)
        # The prompt recipe. Holds the four parts (obs/system/template/tools)
        # plus the per-turn and history hooks; the rollout loop dispatches
        # through it instead of branching on `variant`. When refine is enabled on
        # a non-CH variant, attach the CH refiner bundle onto the resolved spec
        # (CH already carries it, so don't double-attach).
        _spec = spec if spec is not None else resolve_spec(variant, SYSTEM_PROMPT)
        if self.refine_enabled and variant != "CH":
            _spec = attach_refiner(_spec)
        self.spec = _spec
        self.refine_interval = refine_interval
        self.summarize_and_reset = summarize_and_reset
        self.trace_dir = trace_dir
        self.continual = continual
        self.continual_lives = continual_lives
        self.bootstrap_dir = bootstrap_dir
        # Lazy: only build a refiner when refine is enabled (variant=="CH" or
        # refine=True), so other variants don't pull in API clients.
        self.refiner = refiner
        self.refiner_model = refiner_model
        # Explicit escape hatch: run CH with a no-op OfflineRefiner (no teacher
        # required). Tagged via self._ch_real=False so traces / callers can tell
        # this apart from a real teacher-driven CH run. Popped BEFORE
        # super().__init__ so the verifiers base class doesn't choke on it.
        allow_offline_refiner = kwargs.pop("allow_offline_refiner", False)
        # Same-teacher separation override (read in setup_state where the policy
        # model id is observable). Popped here for the same reason.
        self.allow_same_teacher = bool(kwargs.pop("allow_same_teacher", False))
        self._ch_real = False
        if self.refine_enabled and self.refiner is None:
            from nethack_harness.refiner import (
                OfflineRefiner,
                TeacherLLMRefiner,
                resolve_teacher,
            )
            if allow_offline_refiner:
                self.refiner = OfflineRefiner()
            else:
                # Fail loud (CHMisconfigured) if no teacher can be resolved.
                # Thread the resolved base_url + key into the refiner so a key
                # resolved from OPENAI_API_KEY / PI_API_KEY (e.g. GLM via Prime
                # Inference) is actually used, not silently dropped.
                cfg = resolve_teacher(refiner_model)
                self.refiner = TeacherLLMRefiner(
                    model=cfg["model"],
                    base_url=cfg["base_url"],
                    api_key=cfg["api_key"],
                )
                self._ch_real = True
        super().__init__(*args, **kwargs)

    async def setup_state(self, state: vf.State) -> vf.State:
        # Teacher/policy separation guard (whenever the refiner is enabled). The
        # policy model id is injected by verifiers at rollout time (init_state
        # sets state["model"]), so this is the first place we can compare it
        # against the teacher model.
        if self.refine_enabled:
            from nethack_harness.refiner import CHMisconfigured
            policy_model = state.get("model")
            teacher_model = self.refiner_model
            if not policy_model:
                # Policy id genuinely not observable: trust the operator.
                state["ch_separation"] = "operator-asserted"
            elif teacher_model and policy_model == teacher_model:
                state["ch_separation"] = "refused-same-model"
                if not self.allow_same_teacher:
                    raise CHMisconfigured(
                        f"refiner enabled (variant={self.variant!r}) but policy model "
                        f"{policy_model!r} equals the teacher model {teacher_model!r}; "
                        "teacher and policy must differ. "
                        "Pass allow_same_teacher=True to override."
                    )
            else:
                state["ch_separation"] = "separate"

        task: dict = state["task"]
        info: dict = state.get("info") or {}
        # verifiers does not always round-trip the nested "task" dict column, so the
        # per-example tier can be missing here. Fall back to the "info" column
        # (which IS preserved) before the default — the standard full ascension
        # game ("full_nle"), NOT a curriculum tier. Pass tier=<name> to opt into a
        # curriculum tier.
        tier_name: TierName = task.get("tier") or info.get("tier") or "full_nle"
        seed: int = task.get("seed", info.get("seed", random.randint(0, 2**31 - 1)))
        spec = get_tier(tier_name)

        # Game-setup overrides (difficulty/generation knobs, state pokes, custom
        # level). None = vanilla NetHack generation; these are the interface
        # flags that turn the standard game into a customized scenario.
        if tier_name == "curriculum":
            # The compressed full-game curriculum: fixed female-neutral Valkyrie,
            # full vision, and the 1-3 <-> 48-50 <-> planes jump ordering. Owns
            # its own character/vision defaults; the descend/ascend skills drive
            # the cross-branch jumps.
            from nethack_core import CurriculumEnv
            env = CurriculumEnv(max_episode_steps=spec.max_episode_steps)
        else:
            env = NetHackCoreEnv(
                task_name=spec.nle_task,
                max_episode_steps=spec.max_episode_steps,
                des_file=spec.des_file,
                tune=self._setup_tune,
                modify=self._setup_modify,
                level_blob=self._setup_level_blob,
            )
        env.seed(core=seed, disp=seed)
        # NB: bootstrap_character() is currently a stub; once wired up it
        # auto-invokes #attributes and stores role/race/alignment in state.
        # setup_character pins the role for standard tiers (e.g. full_nle);
        # None keeps the engine default. Curriculum envs own their character.
        obs, meta = env.reset(character=self._setup_character)
        from nethack_harness.tools.skills import bootstrap_character
        character = bootstrap_character(env)

        state["env"] = env
        state["character"] = character
        # Continual-harness bookkeeping (no-op when self.continual=False).
        state["_orig_seed"] = int(seed)
        state["_continual_life"] = 1
        state["_continual_lives_left"] = self.continual_lives if self.continual else 0
        state["spec"] = spec
        state["meta"] = meta
        # Standard NLE tiers (full_nle etc.) paint the intro/copyright banner
        # over the top tty rows on every step; the engine-driven CurriculumEnv
        # does not, and owns its own map — so per-turn banner scrubbing is gated
        # to the standard tiers only (see the render path in env_response).
        state["_standard_tier"] = spec.nle_task != "engine"
        state["scout_tiles_seen"] = set()
        state["scout_delta"] = 0
        state["scout_reward_total"] = 0.0
        state["max_dlvl_reached"] = 1
        state["descent_count"] = 0
        state["raw_obs"] = obs
        state["structured_obs"] = shape_observation(obs, character)
        # Drain the intro/startup screen (copyright + version banner) and any
        # message-paging --More-- that NetHack gates behind --More-- on reset.
        # Without this, turn 1's observation carries the banner bleeding into
        # the rendered MAP plus a dangling --More--, garbling the agent's view.
        # We only press MORE/CR (byte 13) while a --More-- is ACTUALLY present
        # (never blindly, so a real [yn]/getlin prompt is left for the agent),
        # and wrap defensively: a drain failure must never break the rollout.
        try:
            more_idx_list = _to_action_indices(env, [13])
            if more_idx_list:
                for _ in range(10):
                    so0 = state["structured_obs"]
                    has_more = (
                        any("--More--" in m for m in (getattr(so0, "messages", None) or []))
                        or _obs_tty_has_more(state["raw_obs"])
                    )
                    if not has_more:
                        break
                    obs, _r0, term0, trunc0, _info0 = env.step(more_idx_list[0])
                    state["raw_obs"] = obs
                    if term0 or trunc0:
                        state["structured_obs"] = shape_observation(obs, character)
                        break
                    state["structured_obs"] = shape_observation(obs, character)
            # The intro/copyright banner is painted over the top tty rows and,
            # in right-offset-map tiers, is never repainted by gameplay — so it
            # bleeds into the rendered MAP. Scrub it from the raw obs before the
            # first observation is shaped/shown. Mutates raw_obs.tty_chars, then
            # re-shape so map_view reflects the cleaned tty.
            _scrub_intro_banner(state["raw_obs"])
            state["structured_obs"] = shape_observation(state["raw_obs"], character)
        except Exception:
            pass
        state["map_detail"] = self.map_detail
        # Track every (x, y) at which `>` was seen on the visible map. Needed
        # because once the player steps ONTO `>`, the @ overlay hides it and
        # extract_visible_features stops finding the tile — without memory,
        # the agent oscillates on/off the stairs without realizing to descend.
        state["_seen_stairs_down"] = set()
        # Wave-2 Track B: visited-frontier memory. Tracks (level_key, (x,y)) →
        # consecutive turns the agent has been within 1 step of this frontier
        # without scout_delta > 0. When the count hits FRONTIER_STUCK_TURNS,
        # the frontier is blacklisted on its level. Blacklist resets on level
        # change (we key by max_dlvl_reached at sighting time). Cleared by
        # `_update_frontier_blacklist` each turn.
        state["_frontier_approach_count"] = {}  # (dlvl, x, y) -> int
        state["_frontier_blacklist"] = {}       # dlvl -> set[(x, y)]
        state["_frontier_prev_dlvl"] = 1
        # Deadlock-breaker flag (Track C reads this; we only set it). Becomes
        # True when all reachable frontiers on the current level are
        # blacklisted AND scout_delta has been 0 for >= NEEDS_HIDDEN_TURNS.
        state["_needs_hidden_passage"] = False
        state["_zero_scout_streak"] = 0
        # Per-turn obs-block flags, declared by the spec's ObsSpec.setup_flags
        # and read by the rendering functions:
        #   _descent_salient -> _descent_status_block (ND/FD)
        #   _e1_obs          -> E1 frontier/coverage/spatial-belief blocks
        #   _e2_obs          -> paint frontier-adjacent unseen tiles on the map
        # Every key is written for all variants (defaulting False) so the
        # rendering reads stay bit-identical to the legacy variant checks.
        _obs_flags = self.spec.obs.setup_flags
        state["_descent_salient"] = _obs_flags.get("_descent_salient", False)
        state["_e1_obs"] = _obs_flags.get("_e1_obs", False)
        state["_e2_obs"] = _obs_flags.get("_e2_obs", False)
        state["last_reward"] = 0.0
        state["terminated"] = False
        state["journal"] = Journal()
        # Variant CH (Continual Harness) component slots. Always initialized so
        # downstream code can read them unconditionally; only populated when
        # variant=="CH" and the Refiner runs (or bootstrap_dir loads them).
        state["_ch_prompt_addendum"] = ""
        state["_ch_subagents"] = {}
        state["_ch_skills"] = {}
        # Pre-pin the tier's description as the agent's objective so the
        # goal stays in every obs (without forcing the model to call
        # pin_objective). For dynamic_subgoal, the proposer pin below
        # overrides this with the LM-proposed objective.
        if spec is not None and getattr(spec, "description", None):
            state["journal"].pin_objective(spec.description)
        if self.sub_lm is not None:
            state["sub_lm"] = self.sub_lm  # used by belief-state distillation
        if self.subgoal_proposer is not None:
            state["subgoal_proposer"] = self.subgoal_proposer

        # Dynamic-subgoal tier: ask the proposer for an episode-specific
        # termination predicate and bolt it onto the spec for env_response
        # to read like any other success_milestone. This is the autoresearch
        # axis: "can an LLM design its own curriculum given the wiki?"
        if tier_name == "dynamic_subgoal":
            from nethack_harness.curriculum.subgoals import compile_predicate, default_proposer
            proposer = state.get("subgoal_proposer") or default_proposer()
            subgoal = proposer.propose(role=character.get("role", "unknown"),
                                        obs=state["structured_obs"])
            milestone = compile_predicate(subgoal.termination_check)
            from dataclasses import replace
            state["spec"] = replace(spec, success_milestone=milestone)
            state["dynamic_subgoal"] = {
                "objective": subgoal.objective,
                "rationale": subgoal.rationale,
                "termination_check": subgoal.termination_check,
            }
            # Pin the objective into the journal so the agent sees it.
            state["journal"].pin_objective(subgoal.objective)

        # Bootstrap I/O for the refiner: if bootstrap_dir is set and a prior
        # snapshot exists for this seed, load the four components in.
        if self.refine_enabled and self.bootstrap_dir:
            try:
                import os, json
                path = os.path.join(self.bootstrap_dir, f"seed{seed}.json")
                if os.path.exists(path):
                    from nethack_harness.refiner import load_components
                    with open(path) as f:
                        load_components(state, json.load(f))
            except Exception:
                pass  # bootstrap failures must never break a rollout

        return state

    async def env_response(self, messages: vf.Messages, state: vf.State) -> vf.Messages:
        # Parse the assistant's tool call from messages[-1].
        # In v0 we expect native function calling (OpenAI tool format).
        assistant_msg = messages[-1]
        # assistant_msg can be a dict (legacy) or a vf.AssistantMessage pydantic object (current).
        if isinstance(assistant_msg, dict):
            tool_calls = assistant_msg.get("tool_calls") or []
        else:
            tool_calls = getattr(assistant_msg, "tool_calls", None) or []
        if not tool_calls:
            # Filter harness-owned skills from the suggestion list — they
            # don't appear in the actual tool schema sent to the model.
            agent_tools = [s for s in list_skills() if s not in ("menu_option", "inventory_item")]
            return [vf.UserMessage(role="user", content="You must call a tool. Available tools: " + ", ".join(agent_tools))]

        # Apply the first tool call (NetHack is turn-based; we ignore multi-call this turn).
        # Verifiers passes tool calls in two shapes depending on version:
        #   old: dict {"function": {"name": ..., "arguments": "..."}}
        #   new: ToolCall pydantic model with flat .name / .arguments
        tc = tool_calls[0]
        # Surface that we dropped the extras so the agent knows only the
        # first call ran — otherwise it might assume all N actions were
        # applied and plan around a stale game state.
        state["_dropped_extra_tool_calls"] = max(0, len(tool_calls) - 1)
        import json
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            skill_name = fn.get("name", tc.get("name", "")) or ""
            raw_args = fn.get("arguments", tc.get("arguments", "{}"))
        else:
            fn = getattr(tc, "function", None)
            skill_name = (getattr(fn, "name", None) if fn is not None else getattr(tc, "name", "")) or ""
            raw_args = getattr(fn, "arguments", None) if fn is not None else getattr(tc, "arguments", "{}")

        # Defensive parsing: small models emit malformed args. Coerce to dict
        # so we never crash on dispatch.
        if raw_args is None or raw_args == "":
            skill_args = {}
        elif isinstance(raw_args, dict):
            skill_args = raw_args
        else:
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    skill_args = parsed
                else:
                    # Model emitted a non-dict (list, scalar, ...). Treat as empty
                    # and let the skill registry surface a friendly error.
                    skill_args = {}
            except (ValueError, TypeError):
                # Malformed JSON — same recovery path.
                skill_args = {}

        env: NetHackCoreEnv = state["env"]

        # Gate hallucinated tool calls: reject any skill the model was NOT given
        # in its tool schema this rollout, instead of dispatching it against the
        # full registry. Without this a model under the `netplay` set (which
        # withholds low-level `move`) can still emit `move(direction=…)` and have
        # it EXECUTE — leaking the primitive into the effective action surface and
        # confounding "hold actions fixed, vary the observation" comparisons.
        # Checked on the ORIGINAL name, before the dir8 rebind below (dir8's
        # compass tools ARE in the exposed set). No NLE step is consumed.
        if self._allowed_skill_names and skill_name not in self._allowed_skill_names:
            avail = ", ".join(sorted(self._allowed_skill_names))
            obs_text = self.spec.turn_template(
                state["structured_obs"], state["journal"], state,
                compact=self.compact_obs,
                journal_max_chars=self.journal_render_max_chars,
            )
            content = compose_user_content(
                obs_text,
                [f"[Tool {skill_name!r} is not available. Call one of: {avail}]"],
            )
            return [vf.UserMessage(role="user", content=content)]

        # dir8 baseline: rewrite north/northeast/.../northwest calls to
        # move(direction=...) so the existing dispatcher handles them.
        _DIR_BIND = {
            "north": "N", "northeast": "NE", "east": "E", "southeast": "SE",
            "south": "S", "southwest": "SW", "west": "W", "northwest": "NW",
        }
        if skill_name in _DIR_BIND:
            skill_args = {"direction": _DIR_BIND[skill_name]}
            skill_name = "move"

        # Variant CH: `run_macro(name=...)` expands a Refiner-registered
        # macro (an ordered list of existing skill calls) into a concatenated
        # SkillResult. Resolution happens here, BEFORE the registry dispatch
        # below, so the rest of env_response sees a normal SkillResult.
        if skill_name == "run_macro":
            from nethack_harness.tools.skills import SkillResult as _SR
            macro_name = (skill_args or {}).get("name", "")
            macro = (state.get("_ch_skills") or {}).get(macro_name)
            if not macro:
                result = _SR(actions=[], feedback=f"Unknown macro: {macro_name!r}", interrupted=True)
            else:
                actions_acc: list = []
                fb_parts: list[str] = []
                interrupted = False
                for step_def in macro:
                    sub_name = step_def.get("skill")
                    sub_args = dict(step_def.get("args") or {})
                    if not sub_name:
                        continue
                    sub_res = skill_registry.call(sub_name, env, state["structured_obs"], **sub_args)
                    if sub_res.journal_op is not None:
                        try:
                            sub_res.journal_op(state["journal"])
                        except Exception:
                            pass
                    actions_acc.extend(sub_res.actions)
                    if sub_res.feedback:
                        fb_parts.append(f"{sub_name}: {sub_res.feedback}")
                    if sub_res.interrupted:
                        interrupted = True
                        break
                result = _SR(
                    actions=actions_acc,
                    feedback=f"[macro:{macro_name}] " + " | ".join(fb_parts),
                    interrupted=interrupted,
                )
        # Code-mode dispatch: if the model called the `code` tool, run the
        # source against the nh namespace and convert its action queue into
        # a SkillResult shape so the rest of env_response can stay unchanged.
        elif self.interface == "code" and skill_name == "code":
            from nethack_harness.tools.code_mode import run_user_code
            from nethack_harness.tools.skills import SkillResult
            source = skill_args.get("source", "")
            cm_result = run_user_code(
                source, env, state["structured_obs"], journal=state.get("journal"),
                raw_obs=state.get("raw_obs"),
            )
            stdout = cm_result.stdout or ""
            err = f"\n[code error: {cm_result.error}]" if cm_result.error else ""
            feedback = (stdout + err).strip() or "(code executed; no stdout)"
            result = SkillResult(actions=cm_result.actions_taken, feedback=feedback)
        else:
            result = skill_registry.call(
                skill_name, env, state["structured_obs"], **skill_args
            )

        # Journal skills: apply the journal op and short-circuit the env step.
        # No NLE turn is consumed; the agent's next prompt reflects the change.
        if result.journal_op is not None:
            journal: Journal = state["journal"]
            feedback = result.journal_op(journal)
            state["scout_delta"] = 0  # no exploration happened
            obs_text = self.spec.turn_template(
                state["structured_obs"], journal, state,
                compact=self.compact_obs,
                journal_max_chars=self.journal_render_max_chars,
            )
            content = compose_user_content(obs_text, [f"[{feedback}]"] if feedback else [])
            return [vf.UserMessage(role="user", content=content)]

        # Capture pre-step scout set size so scout_reward can return a per-step delta
        # rather than a cumulative count. See onboarding/scout_reward.md.
        scout_before = len(state["scout_tiles_seen"])
        # Capture pre-step player (x, y) so we can detect blocked moves.
        pre_pos = None
        try:
            pre_blstats = state["raw_obs"].blstats if state.get("raw_obs") is not None else None
            if pre_blstats is not None:
                pre_pos = (int(pre_blstats[0]), int(pre_blstats[1]))
        except (KeyError, IndexError, TypeError, AttributeError):
            pre_pos = None

        # Skills can return either NLE action enum values (107 == N) or task
        # action-set indices (1 == N for NetHackScore). The underlying gym
        # step expects indices, so convert at the boundary.
        action_indices = _to_action_indices(env, result.actions)

        # Step the underlying env through the action sequence the skill produced.
        # Multi-action skills (autoexplore, move_to) expand into many env.step
        # calls; we halt early on three conditions to give the model a chance
        # to react before walking into a dragon: HP-drop, hostile-in-sight,
        # explicit terminal. This is the "halt on hostile/HP-drop/hunger" item
        # from the project plan.
        total_reward = 0.0
        terminated = truncated = False
        info: dict = {}
        last_obs = state["raw_obs"]
        hp_before = state["structured_obs"].status.get("hitpoints", 0) if state.get("structured_obs") else 0
        halt_reason: Optional[str] = None
        if getattr(result, "pre_executed", False):
            # Closed-loop skill (e.g. explore_and_descend) already stepped the env
            # in its own re-observe loop. Adopt its outcome and skip our step loop.
            total_reward = result.pre_reward
            last_obs = result.final_obs if result.final_obs is not None else state["raw_obs"]
            terminated = bool(result.pre_terminated)
            truncated = bool(result.pre_truncated)
            action_indices = []
        for step_i, action in enumerate(action_indices):
            last_obs, r, terminated, truncated, info = env.step(action)
            total_reward += r
            # Scout reward: count newly-revealed dungeon tiles.
            for (x, y), ch in _iterate_visible_tiles(last_obs):
                if ch not in (b" ", b"\x00"):
                    state["scout_tiles_seen"].add((state["max_dlvl_reached"], x, y))
            if terminated or truncated:
                break
            # Status-aware halt: check after each step (cheap — just blstats).
            # Only enabled for multi-step skills (>=4 actions in a single tool
            # call) so single-key skills aren't penalized by the overhead.
            if len(action_indices) >= 4 and step_i + 1 < len(action_indices):
                halt_reason = _check_halt_condition(last_obs, hp_before)
                if halt_reason:
                    break
                # Also halt if a y/n / menu prompt opened mid-sequence — the
                # remaining action indices would be consumed as keystroke
                # answers to the prompt rather than continuing the intended
                # action sequence (e.g. autoexplore step 16 would answer
                # "Really attack?" as 'n' instead of moving NE).
                msg_bytes = last_obs.get("message") if isinstance(last_obs, dict) else None
                if msg_bytes is not None:
                    msg = bytes(msg_bytes).split(b"\x00", 1)[0].decode("ascii", errors="replace")
                    if "[yn" in msg or "--More--" in msg:
                        halt_reason = "prompt opened mid-sequence"
                        break

        scout_after = len(state["scout_tiles_seen"])
        state["scout_delta"] = scout_after - scout_before
        # Accumulate cumulative scout reward: the rubric scores once at end of
        # rollout, so a per-step `scout_delta` alone would only reflect the
        # final step. Sum here so scout_reward can report total exploration.
        state["scout_reward_total"] += state["scout_delta"] / 1000.0

        state["raw_obs"] = last_obs
        state["structured_obs"] = shape_observation(last_obs, state["character"])
        # Auto-dismiss any menu/inventory_prompt that's still open. Menus are
        # mechanical (--More--, level-up choice picker, multi-page item lists)
        # and were a huge time-sink for the LM agent: Qwen3.5-9B spent 42% of
        # turns on menu_option / inventory_item calls (often nonsensical) before
        # this hook. By auto-pressing ESC, the harness owns the menu-navigation
        # responsibility and the agent sees a clean post-menu observation on the
        # next turn. The `eat`/`quaff`/`read` skills now bundle item selection
        # in-skill, so intentional inventory prompts also resolve here.
        dismissed = 0
        esc_idx_list = _to_action_indices(env, [27])
        more_idx_list = _to_action_indices(env, [13])
        y_idx_list = _to_action_indices(env, [ord('y')])
        n_idx_list = _to_action_indices(env, [ord('n')])
        esc_action = esc_idx_list[0] if esc_idx_list else (more_idx_list[0] if more_idx_list else None)
        y_action = y_idx_list[0] if y_idx_list else esc_action
        n_action = n_idx_list[0] if n_idx_list else esc_action
        for _ in range(8):
            so = state["structured_obs"]
            yn = getattr(so, "yn_prompt", None)
            # Detect --More-- prompts in the message buffer too — they consume
            # the next keystroke, which would otherwise eat the model's
            # intended action. MORE/CR (13) acknowledges them.
            has_more = any("--More--" in m for m in (so.messages or [])) or _obs_tty_has_more(last_obs)
            if so.menu is None and so.inventory_prompt is None and yn is None and not has_more:
                break
            if yn is not None:
                ans = yn["answer"]
                action = y_action if ans == "y" else (n_action if ans == "n" else esc_action)
            elif has_more:
                # MORE prompts want CR/space, not ESC.
                action = more_idx_list[0] if more_idx_list else esc_action
            else:
                action = esc_action
            if action is None:
                break
            last_obs, _r, t2, tr2, _info = env.step(action)
            terminated = terminated or t2
            truncated = truncated or tr2
            state["raw_obs"] = last_obs
            state["structured_obs"] = shape_observation(last_obs, state["character"])
            dismissed += 1
            if terminated or truncated:
                break
        if dismissed:
            halt_reason = (halt_reason or "") + (f" menu auto-dismissed x{dismissed}" if not halt_reason else f" / menu auto-dismissed x{dismissed}")
            halt_reason = halt_reason.lstrip()
        state["last_reward"] = total_reward
        state["terminated"] = terminated or truncated
        # Refiner: on terminal, persist the refined components for the
        # next rollout (if bootstrap_dir is configured).
        if state["terminated"] and self.refine_enabled:
            _ch_save_bootstrap(self, state)
        # Continual harness mode: if the agent died (not ascended), and lives
        # remain, auto-reseed and reset the NLE env so the chat session
        # continues into a new game. Journal + belief state survive across
        # lives — this is the "memory persists; episodes don't" pattern.
        if (
            self.continual
            and (terminated or truncated)
            and not state.get("ascended", False)
            and state.get("_continual_lives_left", self.continual_lives) > 0
        ):
            try:
                _continual_reset(state, env, self)
                terminated = truncated = False
                state["terminated"] = False
            except Exception as e:
                # Best-effort: if reset fails, end the rollout normally.
                state["_continual_error"] = repr(e)
        # BALROG-style progression score (informational; not in rubric).
        # Tracks deepest (DL, XL) achieved as an empirical-ish P(ascend).
        from nethack_harness.prompt.balrog import progression_score
        s = state["structured_obs"].status
        state["balrog_progression"] = progression_score(
            state["max_dlvl_reached"], s.get("experience_level", 1)
        )
        # Death/ascension detection from the game state, not raw NLE termination flag.
        _detect_terminal_outcome(last_obs, state)
        # Robust death fallback: the text-marker scan above misses most deaths
        # because the death / "Do you want your possessions identified?" screen is
        # auto-dismissed inside closed-loop skills (explore_and_descend) before
        # env_response ever sees it — so `died` was only catching ~1 in 7 deaths.
        # NLE's terminated flag is authoritative: a game that NLE ended and that
        # we did NOT detect as an ascension is, at these depths, a death. (Milestone
        # success sets state["terminated"] separately, AFTER this block, so it can't
        # be confused for a death here.)
        if terminated and not state["ascended"] and not state["died"]:
            state["died"] = True
            state.setdefault("death_dlvl", state.get("max_dlvl_reached", 1))
        # Milestone-driven success: if the tier's success_milestone fires, we
        # treat the rollout as won and let success_reward pay out.
        spec = state.get("spec")
        if spec is not None and getattr(spec, "success_milestone", None) is not None:
            if spec.success_milestone.check(last_obs, state):
                state["succeeded"] = True
                state["terminated"] = True

        # Belief-state distillation (Track B v0.3): two trigger conditions.
        # 1) Level transition: summarize the prior level into the journal.
        # 2) Periodic (every BELIEF_STATE_INTERVAL turns): summarize the
        #    recent journal into a compact "belief_state" note so history-
        #    compaction can drop turns >100 without losing the LM's mental
        #    model. Survey rec #3.
        new_dlvl = state["structured_obs"].status.get("depth", 1)
        if new_dlvl > state["max_dlvl_reached"]:
            _maybe_distill(state, prior_dlvl=state["max_dlvl_reached"])
            # Count the descent here so descent_reward can read a cumulative
            # tally at end-of-rollout. (The rubric only fires score_rollout
            # once, so a per-step compare would lose every transition except
            # the last.)
            state["descent_count"] = state.get("descent_count", 0) + (new_dlvl - state["max_dlvl_reached"])
            state["max_dlvl_reached"] = new_dlvl  # update AFTER computing the level delta

        # Wave-2 Track B: update visited-frontier memory + deadlock flag.
        try:
            _update_frontier_blacklist(state)
        except Exception:
            pass

        state["turn_count"] = state.get("turn_count", 0) + 1
        if self.belief_state_interval > 0 and state["turn_count"] > 0 and state["turn_count"] % self.belief_state_interval == 0:
            _maybe_belief_state_summary(state)

        # Move-blocked detection: `move(direction=...)` always reports "Moved
        # S." even when the action bumped a wall. The model can't tell from
        # feedback whether the step succeeded. Compare pre/post player (x, y)
        # from blstats; if a single-step move kept us in place, override the
        # feedback so the model knows to pick a different direction.
        if skill_name == "move" and len(action_indices) == 1 and pre_pos is not None and not terminated and not truncated:
            try:
                from nethack_harness.tools.skills import SkillResult as _SR
                post_blstats = last_obs.blstats if hasattr(last_obs, "blstats") else last_obs.get("blstats")
                if post_blstats is not None:
                    post_pos = (int(post_blstats[0]), int(post_blstats[1]))
                    if post_pos == pre_pos:
                        result = _SR(
                            actions=result.actions,
                            feedback=f"Move blocked at {pre_pos}: wall or obstacle in {skill_args.get('direction', '?')}. Pick a different direction or `search` if you suspect a hidden door.",
                            interrupted=result.interrupted,
                        )
            except (KeyError, IndexError, TypeError, AttributeError):
                pass

        # Attack-outcome detection: replace the generic "Moved W." feedback
        # with hit/miss/kill info pulled from the NLE message buffer. The
        # model doesn't otherwise know whether its swing landed.
        if skill_name == "attack":
            try:
                from nethack_harness.tools.skills import SkillResult as _SR
                msg_bytes = last_obs.message if hasattr(last_obs, "message") else last_obs.get("message")
                if msg_bytes is not None:
                    msg = bytes(msg_bytes).split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
                    if msg:
                        outcome = None
                        msg_l = msg.lower()
                        if "you kill" in msg_l or "is killed" in msg_l:
                            outcome = f"Killed: {msg}"
                        elif "you hit" in msg_l or "you destroy" in msg_l:
                            outcome = f"Hit: {msg}"
                        elif "you miss" in msg_l:
                            outcome = f"Missed: {msg}"
                        elif "nothing here" in msg_l or "no monster" in msg_l:
                            outcome = f"No target: {msg}"
                        if outcome:
                            result = _SR(actions=result.actions, feedback=outcome, interrupted=result.interrupted)
            except (KeyError, IndexError, TypeError, AttributeError):
                pass

        # ---- Wave-2: position-stuck deadlock breaker -------------------
        # Diagnosis (experiment_log.md Wave-2): exploration skills can wedge.
        # A scripted `find_and_descend` loop on seed 22 froze at pos (63,6)
        # with the in-game clock stuck at T:51 for 90+ turns: A* kept choosing
        # a 1-step "far frontier" whose first step was a no-op (an adjacent
        # closed/locked door `+` that walking-into does nothing). Real LM
        # rollouts hit the same wedge and oscillate until they starve.
        #
        # Fix: when a movement/exploration skill leaves the player at the SAME
        # (x,y) AND the in-game turn counter did not advance, count it. After
        # 2 such no-progress calls, auto-KICK an adjacent closed door `+` to
        # break the wedge (or, if none, surface a strong search-or-redirect
        # hint). This runs for every variant — it's a pure correctness fix.
        _EXPLORE_SKILLS = {"autoexplore", "find_and_descend", "move_to", "move"}
        stuck_hint: Optional[str] = None
        if skill_name in _EXPLORE_SKILLS and pre_pos is not None and not (terminated or truncated):
            try:
                post_blstats = last_obs.blstats if hasattr(last_obs, "blstats") else last_obs.get("blstats")
                post_pos = (int(post_blstats[0]), int(post_blstats[1])) if post_blstats is not None else None
            except (KeyError, IndexError, TypeError, AttributeError):
                post_pos = None
            # No-progress = the in-game clock did not advance. This catches both
            # "bumped a wall" (pos same) AND the false-frontier oscillation
            # (pos toggles between two adjacent corridor tiles whose only
            # "unexplored" neighbor is actually solid rock, so the game clock
            # never moves and no new tiles are revealed). Clock-frozen is the
            # reliable signal: a real exploration step always advances T.
            cur_time = state["structured_obs"].status.get("time")
            prev_time = state.get("_last_stuck_time")
            clock_frozen = (prev_time is not None and cur_time == prev_time)
            # Also track revealed-tile count: if the scout set didn't grow,
            # the call revealed nothing new.
            no_new_tiles = state.get("scout_delta", 0) == 0
            no_progress = clock_frozen and no_new_tiles
            if no_progress:
                state["_stuck_count"] = state.get("_stuck_count", 0) + 1
            else:
                state["_stuck_count"] = 0
            state["_last_stuck_time"] = cur_time
            if state.get("_stuck_count", 0) >= 2:
                # Find an adjacent closed door `+` to kick.
                adj = getattr(state["structured_obs"], "adjacent", None) or {}
                door_dir = None
                for d, tile in adj.items():
                    if tile and (tile == "+" or tile.startswith("+")):
                        door_dir = d
                        break
                if door_dir is not None:
                    # Auto-kick: KICK command then the direction key. Step it.
                    try:
                        from nethack_harness.tools.skills import _DIRECTION_KEYS  # type: ignore
                    except Exception:
                        _DIRECTION_KEYS = None
                    _DKEY = {"N": ord("k"), "S": ord("j"), "E": ord("l"), "W": ord("h"),
                             "NE": ord("u"), "NW": ord("y"), "SE": ord("n"), "SW": ord("b")}
                    from nethack_core import actions as _nh
                    kick_cmd = int(_nh.Command.KICK)
                    kick_seq = _to_action_indices(env, [kick_cmd]) + _to_action_indices(env, [_DKEY.get(door_dir, ord("."))])
                    for ka in kick_seq:
                        last_obs, _kr, kt, ktr, _ki = env.step(ka)
                        terminated = terminated or kt
                        truncated = truncated or ktr
                        if terminated or truncated:
                            break
                    state["raw_obs"] = last_obs
                    state["structured_obs"] = shape_observation(last_obs, state["character"])
                    state["_stuck_count"] = 0
                    stuck_hint = (
                        f"[deadlock-breaker: stuck at {pre_pos}; auto-kicked the "
                        f"closed door to {door_dir}. If it didn't open, kick again "
                        f"or pick a different exploration target.]"
                    )
                else:
                    # No door to kick and the level's visible frontiers are
                    # all false (adjacent only to solid rock) — the genuine
                    # exit is a HIDDEN passage. Auto-search in place to reveal
                    # it, escalating count the longer we're wedged. NLE caps a
                    # search run, so this is safe. This is what unwedges the
                    # seed-22 corridor pocket (all frontiers border rock).
                    from nethack_core import actions as _nh
                    search_idx = _to_action_indices(env, [int(_nh.Command.SEARCH)])
                    n_search = min(10 * state.get("_stuck_count", 2), 30)
                    if search_idx:
                        for _ in range(n_search):
                            last_obs, _sr, stt, str_, _si = env.step(search_idx[0])
                            terminated = terminated or stt
                            truncated = truncated or str_
                            if terminated or truncated:
                                break
                        state["raw_obs"] = last_obs
                        state["structured_obs"] = shape_observation(last_obs, state["character"])
                    state["_stuck_count"] = 0
                    stuck_hint = (
                        f"[deadlock-breaker: exploration wedged at {pre_pos} (all "
                        f"reachable frontiers border solid rock). Auto-searched "
                        f"{n_search}x for a hidden passage. If still no new exit, "
                        f"`move_to` a DIFFERENT visible tile or `search` more — the "
                        f"way down is likely behind a hidden wall.]"
                    )
            # If the auto-kick/search advanced the game state to a terminal
            # outcome (e.g. starved mid-search), re-run detection so death is
            # attributed and the rollout ends cleanly.
            if terminated or truncated:
                state["terminated"] = True
                _detect_terminal_outcome(last_obs, state)
        # ----------------------------------------------------------------

        # Autoexplore-loop detection: when autoexplore returns "short" feedback
        # repeatedly (frontier shrunk to 1-2 step paths near level edges), the
        # model often spam-calls it ignoring the tail hint. After N consecutive
        # short trips, emit a stronger interrupt hint at the TOP of the obs.
        # Trace 9071d001 showed 66 autoexplore calls with 7-long runs ignoring
        # in-skill tail tips.
        loop_hint: Optional[str] = None
        if skill_name == "autoexplore" and result.feedback and "short" in result.feedback:
            state["consecutive_short_autoexplore"] = state.get("consecutive_short_autoexplore", 0) + 1
            n = state["consecutive_short_autoexplore"]
            if n >= 3:
                loop_hint = (
                    f"[autoexplore-loop: {n} short trips in a row. "
                    "Switch tactic: `search` adjacent walls, or `move_to(x,y)` "
                    "a specific feature, or pick a direction with `move`.]"
                )
        else:
            state["consecutive_short_autoexplore"] = 0

        # Re-scrub the intro/copyright banner before rendering. Each env.step
        # re-carries the banner in the still-unexplored top tty rows, and in the
        # right-offset-map standard tiers it is never repainted by gameplay — so
        # the one-time scrub in setup_state is not enough: it must be reapplied
        # to the freshly-stepped obs every turn or it bleeds into the rendered
        # MAP (which reads the raw tty). Gated to standard tiers; defensive
        # (never raises). Mutates raw_obs.tty_chars, then re-shape so the map
        # view reflects the cleaned tty.
        if state.get("_standard_tier", True):
            _scrub_intro_banner(state["raw_obs"])
            state["structured_obs"] = shape_observation(state["raw_obs"], state["character"])

        # Build the per-turn user message from the spec's turn template.
        obs_text = self.spec.turn_template(
            state["structured_obs"], state["journal"], state,
            compact=self.compact_obs,
            journal_max_chars=self.journal_render_max_chars,
        )
        prefix_parts = []
        # Per-turn hooks declared by the spec (P self-refinement directive; CH
        # refiner + sub-agent triggers). Each mutates prefix_parts/state in
        # place. Variants with no hooks (the default) skip this entirely.
        for hook in self.spec.turn_hooks:
            hook(self, state, prefix_parts)
        if stuck_hint:
            prefix_parts.append(stuck_hint)
        if loop_hint:
            prefix_parts.append(loop_hint)
        if halt_reason:
            prefix_parts.append(f"[autohalt: {halt_reason}]")
        dropped = state.get("_dropped_extra_tool_calls", 0)
        if dropped:
            prefix_parts.append(
                f"[multi-tool warning: only the first of {dropped+1} tool "
                "calls was applied. NetHack is turn-based; emit ONE tool "
                "call per turn.]"
            )
            state["_dropped_extra_tool_calls"] = 0
        if result.feedback:
            prefix_parts.append(f"[{result.feedback}]")
        content = compose_user_content(obs_text, prefix_parts)
        # Per-turn trace (NDJSON) for replay/debugging. No-op when trace_dir
        # is unset; never raises.
        _write_trace_entry(
            self, state, assistant_msg, tool_calls,
            action_indices, total_reward, content_to_text(content), obs_content=content,
        )
        return [vf.UserMessage(role="user", content=content)]

    async def is_completed(self, state: vf.State) -> bool:
        # Game-over (death/ascension/NLE truncation) ends the rollout.
        if bool(state.get("terminated")):
            return True
        # Also honor the verifiers per-rollout LM-turn cap (`max_turns`). Without
        # this, the override silently bypassed the base class's
        # `max_turns_reached`, so `max_turns` was a no-op and rollout length was
        # governed solely by the tier's `max_episode_steps` (in-game NLE steps).
        # OR-ing it in makes `max_turns` an effective per-rollout LM-call cap.
        if getattr(self, "max_turns", -1) and self.max_turns > 0:
            if await self.max_turns_reached(state):
                state["is_truncated"] = True
                return True
        return False

    async def get_prompt_messages(self, state: vf.State):
        """Override the verifiers default to compact older user-message content
        (i.e. our prior turn observations) before sending to the LM. This is
        the biggest token-bill win: chat history grew linearly in turns,
        re-sending the full tty grid (~25k tok/turn) every single time. After
        compaction:
          * last K=5 turns: full fidelity
          * turns K..100: replaced with a one-line "[turn N: <summary>]"
          * turns >100: dropped entirely
        Mirrors SWE-agent's "elide all but last 5" and Glyphbox's 10/100
        thresholds (see docs/PROMPTING_SURVEY.md).
        """
        messages = await super().get_prompt_messages(state)
        # Compaction is universal (the biggest token-bill win).
        messages = _compact_chat_history(messages, keep_full=self.history_keep_full, drop_after=self.history_drop_after)
        # Summarize-and-reset is an orthogonal knob (variant R sets it): drop
        # everything prior to the most-recent belief checkpoint.
        if self.summarize_and_reset:
            messages = _drop_before_last_belief(messages, state)
        # Spec-declared history transforms (CH appends the Refiner addendum +
        # macro list to the system message). Empty for most variants.
        for transform in self.spec.history_transforms:
            messages = transform(self, messages, state)
        # Strict OpenAI-compatible endpoints (e.g. Prime Inference / Qwen3.5)
        # reject a request whose history contains an assistant message with
        # content=None and no tool_calls -- which is exactly what a "thinking"
        # model emits when a turn is pure reasoning_content (HTTP 422:
        # "content is required unless an assistant message includes tool_calls").
        # Coerce such messages to a non-null content so the rollout can proceed.
        messages = _sanitize_assistant_content(messages)
        return messages

    def update_tool_args(self, tool_args: dict, messages, state) -> dict:
        """
        Required by StatefulToolEnv. We dispatch tool calls manually inside
        `env_response` (because each skill has a custom signature involving
        the env handle + structured observation), so this hook is a no-op:
        we never let the base class's `call_tool()` route get used.
        """
        return tool_args



# ---------- frontier blacklist (kept here: tests monkeypatch these on the nethack module) ----------

def _obs_tty_has_more(obs) -> bool:
    """True if a --More-- prompt is visible on the top tty rows.

    Works whether obs is a dict (``obs["tty_chars"]``) or a CoreObservation
    (attribute access ``obs.tty_chars``). Mirrors the tty-row detector used in
    nethack_harness/tools/skills.py. Never raises — a detector failure must
    never break a rollout.
    """
    try:
        tty = obs.get("tty_chars") if isinstance(obs, dict) else getattr(obs, "tty_chars", None)
        if tty is None:
            return False
        return any(b"--More--" in bytes(int(c) for c in row) for row in tty[:3])
    except Exception:
        return False


# Substrings that identify the NetHack startup/intro banner. NLE paints this
# copyright/version art over the top tty rows on reset, and — in tiers whose
# map is offset to the right so the player never walks over those cells — the
# banner is never repainted by gameplay. It then bleeds into the rendered MAP
# (render_map_view reads the raw tty), garbling the agent's first observations.
_INTRO_BANNER_MARKERS = (b"Copyright", b"Stichting", b"Version 3.6", b"See license")


def _scrub_intro_banner(obs) -> None:
    """Strip the stale intro/copyright banner from a CoreObservation's tty.

    The authoritative dungeon map lives in ``obs.chars`` (one row above the
    matching tty row, same columns). Wherever a top tty cell holds banner art
    that the clean ``chars`` plane reports as blank, we overwrite it with a
    space. Real map glyphs (which match ``chars`` exactly) are left untouched.
    Mutates ``obs.tty_chars`` in place. Never raises — a scrub failure must
    never break a rollout.
    """
    try:
        tty = getattr(obs, "tty_chars", None)
        chars = getattr(obs, "chars", None)
        if tty is None or chars is None:
            return
        # Only act when a banner signature is actually present on the top rows,
        # so we never disturb a normal observation.
        top = bytes(int(c) for r in range(min(6, tty.shape[0])) for c in tty[r])
        if not any(m in top for m in _INTRO_BANNER_MARKERS):
            return
        space = ord(" ")
        n_cols = min(tty.shape[1], chars.shape[1])
        # tty row r (1..21) corresponds to chars row r-1, same columns.
        for r in range(1, min(tty.shape[0], chars.shape[0] + 1)):
            for x in range(n_cols):
                if tty[r, x] != chars[r - 1, x] and chars[r - 1, x] == space:
                    tty[r, x] = space
    except Exception:
        pass


def _iterate_visible_tiles(obs):
    """Yield ((x, y), char) for currently-visible map tiles."""
    chars = obs.chars  # (21, 79)
    for y in range(chars.shape[0]):
        for x in range(chars.shape[1]):
            yield (x, y), bytes([int(chars[y, x])])


# ----- Wave-2 Track B: visited-frontier memory + deadlock-breaker -----
#
# Knobs (kept module-level so tests can monkeypatch):
FRONTIER_STUCK_TURNS = 3      # adjacency turns w/o new tiles before blacklist
FRONTIER_APPROACH_RADIUS = 1  # Chebyshev distance counting as "approached"
NEEDS_HIDDEN_TURNS = 5        # zero-scout streak that triggers needs-hidden


def _update_frontier_blacklist(state: dict) -> None:
    """Per-turn maintenance of the visited-frontier memory.

    Rules:
      * Reset blacklist + counters when max_dlvl_reached changes (per-level
        memory is what we want — a "stuck" frontier on L1 isn't stuck on L2).
      * Walk all current-level frontiers. For each frontier within
        FRONTIER_APPROACH_RADIUS of the player, increment its consecutive
        no-progress count iff `scout_delta == 0` this turn. Any turn that
        revealed new tiles resets all counts (we're making progress somehow).
      * When a frontier's count hits FRONTIER_STUCK_TURNS, add it to the
        per-level blacklist.
      * Set `_needs_hidden_passage` when every reachable frontier is
        blacklisted AND we've had `_zero_scout_streak >= NEEDS_HIDDEN_TURNS`.
        Cleared when scout_delta > 0 (the search/kick worked).

    Track C reads `state["_needs_hidden_passage"]`; we only set it here.
    """
    from nethack_harness.navigation.pathfinding import find_frontiers
    raw = state.get("raw_obs")
    if raw is None or not hasattr(raw, "chars") or not hasattr(raw, "blstats"):
        return
    chars = raw.chars
    blstats = raw.blstats
    px, py = int(blstats[0]), int(blstats[1])
    cur_dlvl = int(state.get("max_dlvl_reached", 1))
    prev_dlvl = state.get("_frontier_prev_dlvl", cur_dlvl)
    if cur_dlvl != prev_dlvl:
        # Level changed — clear per-level state and exit.
        state["_frontier_approach_count"] = {}
        state["_frontier_blacklist"].pop(prev_dlvl, None)
        state["_frontier_prev_dlvl"] = cur_dlvl
        state["_needs_hidden_passage"] = False
        state["_zero_scout_streak"] = 0
        return
    scout_delta = int(state.get("scout_delta", 0) or 0)
    if scout_delta > 0:
        # Real progress — wipe counters AND clear hidden-passage flag.
        state["_frontier_approach_count"] = {}
        state["_needs_hidden_passage"] = False
        state["_zero_scout_streak"] = 0
        return
    state["_zero_scout_streak"] = int(state.get("_zero_scout_streak", 0)) + 1
    blacklist_for_lvl = state["_frontier_blacklist"].setdefault(cur_dlvl, set())
    # Use legacy (loose) predicate for blacklist accounting so we don't miss
    # nominal frontiers — strict predicate culls them at pick time anyway.
    frontiers = find_frontiers(chars, blacklist=None, strict=False)
    approach = state["_frontier_approach_count"]
    for fx, fy in frontiers:
        if (fx, fy) in blacklist_for_lvl:
            continue
        cheb = max(abs(fx - px), abs(fy - py))
        if cheb <= FRONTIER_APPROACH_RADIUS:
            key = (cur_dlvl, fx, fy)
            approach[key] = approach.get(key, 0) + 1
            if approach[key] >= FRONTIER_STUCK_TURNS:
                blacklist_for_lvl.add((fx, fy))
    # Recompute reachable-frontier set (strict + blacklisted).
    open_frontiers = find_frontiers(chars, blacklist=blacklist_for_lvl, strict=True)
    if not open_frontiers and state["_zero_scout_streak"] >= NEEDS_HIDDEN_TURNS:
        state["_needs_hidden_passage"] = True
    # Expose the current-level blacklist on the env so the in-skill autoexplore
    # picker can consume it without needing the full verifiers state dict (which
    # it doesn't have access to).
    try:
        env_obj = state.get("env")
        if env_obj is not None:
            env_obj.frontier_blacklist_current = set(blacklist_for_lvl)
    except Exception:
        pass




def _build_task_dataset(tier: Optional[TierName], n_examples: int, seed_base: int, explicit_seeds: Optional[list] = None, system_prompt: Optional[str] = None) -> Dataset:
    """Each row is one starting condition.

    If `explicit_seeds` is provided (list of ints), each row uses one of
    those NLE seeds (cycling through), and n_examples is overridden by the
    list length. Use this to pin known-easy seeds for evaluation.

    `system_prompt` defaults to this module's SYSTEM_PROMPT (honoring any
    NETHACK_HARNESS overlay); load_environment passes the spec's prompt.
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    rng = random.Random(seed_base)
    if tier is None:
        tiers = list_tiers()
    else:
        tiers = [tier]
    rows = []
    if explicit_seeds is not None:
        n_examples = len(explicit_seeds)
    for i in range(n_examples):
        t = rng.choice(tiers)
        spec = get_tier(t)
        seed_val = (int(explicit_seeds[i]) if explicit_seeds is not None
                    else rng.randint(0, 2**31 - 1))
        rows.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: {spec.description}\nSuccess: {spec.success_criterion}\n\nBegin."},
            ],
            "task": {"tier": t, "seed": seed_val},
            "info": {"tier": t, "spec_description": spec.description},
        })
    return Dataset.from_list(rows)



def load_environment(
    tier: Optional[str] = "full_nle",
    n_examples: int = 256,
    seed: int = 0,
    max_turns: int = 200,
    interface: str = "skill",
    # --- Game-setup overrides: the difficulty / generation knobs from our
    # engine interface. All default to None = vanilla NetHack. Pass these to
    # customize the game instead of the standard ascension run; e.g.
    #   tune={"vision_radius": 5, "mob_spawn": 2, "room_density": 0.3}  # difficulty/generation
    #   modify={"hp": 200, "max_hp": 200, "gold": 1000}                 # starting stat pokes
    #   level_blob="path/to/level.blob"                                  # custom starting level
    # `tune` keys come from EngineEnv.tune.catalog(); `modify` keys are the
    # whitelisted state setters (hp/max_hp/gold/xp_level/hunger).
    tune: Optional[dict] = None,
    modify: Optional[dict] = None,
    level_blob: Optional[str] = None,
    # Pin the character for standard tiers, e.g. "Val-hum-neu-fem". None =
    # engine default (random roll). Ignored by curriculum tiers (own character).
    character: Optional[str] = None,
    sub_lm=None,
    subgoal_proposer=None,
    compact_obs: bool = True,
    history_keep_full: int = 5,
    history_drop_after: int = 100,
    belief_state_interval: int = 25,
    journal_render_max_chars: int = 2000,
    variant: str = "B1",
    refine_interval: int = 20,
    summarize_and_reset: bool = False,
    trace_dir: Optional[str] = None,
    continual: bool = False,
    continual_lives: int = 5,
    refiner: Any = None,
    refiner_model: Optional[str] = None,
    bootstrap_dir: Optional[str] = None,
    refine: bool = False,
    **kwargs: Any,
) -> vf.Environment:
    """
    Entrypoint used by `vf-eval` and prime-rl.

    Args:
        tier: curriculum tier name, or None for uniform sampling across all tiers.
        n_examples: dataset size for training rollouts / evaluation.
        seed: RNG seed for which (tier, episode-seed) pairs get sampled.
        max_turns: per-rollout turn cap (LM turns, not in-game turns).
        interface: "skill" (default — one tool per skill, OpenAI function-calling)
            or "code" (a single `code` tool that runs sandboxed Python against
            an `nh` namespace; the Track B / RLM-research path).
        compact_obs: enable per-turn observation compaction (strip blank tty rows,
            glyph-run encoding, inventory diff). Default True; set False for raw v0.0.15
            rendering (replay/debugging).
        history_keep_full: number of most-recent turns kept at full fidelity in the
            LM prompt (older turns get a one-line summary or are dropped).
        history_drop_after: turns older than this distance are dropped behind a
            single elision marker.
        belief_state_interval: every N turns, SubLM.summarize is invoked and the
            result added to the journal as belief_state:tN. Set to 0 to disable.
        journal_render_max_chars: soft cap on per-turn journal block size; older
            non-belief-state notes get elided when over the cap.
        variant: obs/skill-structure variant for wave-1 experiments.
            "B1" (default) = current shipping behavior, no override.
            "P" = Continual Harness adaptation (arXiv:2605.09998): every
            `refine_interval` turns, inject a self-refinement directive
            asking the agent to revise its pinned objective and/or record
            a lesson note. Journal ops short-circuit the NLE step, so
            refinement is free game-turn-wise.
        refine_interval: cadence for variant=P self-refinement turns
            (default 20). Set to 0 to disable even when variant="P".
        summarize_and_reset: variant=R toggle. When True, get_prompt_messages
            drops every chat turn older than the most recent belief_state
            checkpoint. Pair with belief_state_interval > 0.
        trace_dir: if set, env_response writes per-turn NDJSON capturing
            raw_grid, structured_obs, rendered_user_message, assistant_message,
            tool_calls, action, reward, dlvl, hp. One file per rollout under
            <trace_dir>/<run_id>.ndjson. Off by default.
        continual: when True, the env auto-resets the underlying NLE on death
            and continues the same chat session, preserving journal + belief
            state. Implements the continual-harness mode (separate from
            variant=P's mid-rollout refinement).
        continual_lives: cap on auto-resets within a single rollout
            (default 5). Ignored unless continual=True.
        refine: when True, attach the full Continual-Harness teacher Refiner
            machinery (periodic teacher-LLM edits to the agent's prompt /
            sub-agents / skill-macros / journal, plus the run_macro tool) onto
            whatever `variant` (obs format) is selected — decoupling the refiner
            from the canonical-ASCII CH variant. e.g. variant="JSON",
            refine=True gives JSON observations PLUS the teacher refiner.
            variant="CH" always implies refine=True.
    """
    explicit_seeds = kwargs.pop("explicit_seeds", None)
    # NETHACK_HARNESS overlay: mutates SYSTEM_PROMPT (consumed by _build_task_dataset
    # below) plus returns a HarnessConfig used to filter tools / re-weight rewards.
    # No-op when the env var is unset → bit-identical default behavior.
    import sys as _sys
    _overlay_cfg = _harness_overlay.apply_overlay(_sys.modules[__name__])
    # Resolve the prompt recipe AFTER the overlay so the spec carries the
    # (possibly-overlaid) system prompt. SYSTEM_PROMPT here is this module's
    # global, which apply_overlay just mutated in place.
    spec = resolve_spec(variant, SYSTEM_PROMPT)
    # Decouple the teacher refiner from the obs format: when refine=True on a
    # non-CH variant, attach the CH refiner bundle (hooks + system inject +
    # run_macro tool) onto the resolved spec so the tool gets exposed below and
    # the env's spec carries the refiner hooks. (CH already carries it.)
    if bool(refine) and variant != "CH":
        spec = attach_refiner(spec)
    dataset = _build_task_dataset(
        tier, n_examples, seed, explicit_seeds=explicit_seeds,
        system_prompt=spec.system_prompt,
    )
    _reward_funcs = _harness_overlay.apply_reward_weights(
        [scout_reward, descent_reward, success_reward, ascension_reward], _overlay_cfg,
    )
    rubric = vf.Rubric(funcs=_reward_funcs)

    if interface == "skill":
        tool_callables = _build_skill_adapter_callables(
            skill_set=spec.tools.skill_set or kwargs.pop("skill_set", "full")
        )
        # Spec-declared extra tools (e.g. CH's run_macro adapter).
        for make_tool in spec.tools.extra_tools:
            tool_callables.append(make_tool())
        tool_callables = _harness_overlay.filter_tool_callables(tool_callables, _overlay_cfg)
    elif interface == "code":
        tool_callables = [_code_tool_adapter()]
    else:
        raise ValueError(f"Unknown interface={interface!r}; expected 'skill' or 'code'.")
    # The exact set of tool names offered to the model — used to gate
    # hallucinated tool calls in env_response (see allowed_skill_names).
    _allowed_skill_names = {getattr(t, "__name__", "") for t in tool_callables} - {""}

    return NetHackVerifiersEnv(
        dataset=dataset,
        rubric=rubric,
        tools=tool_callables,
        max_turns=max_turns,
        interface=interface,
        sub_lm=sub_lm,
        subgoal_proposer=subgoal_proposer,
        compact_obs=compact_obs,
        history_keep_full=history_keep_full,
        history_drop_after=history_drop_after,
        belief_state_interval=belief_state_interval,
        journal_render_max_chars=journal_render_max_chars,
        variant=variant,
        spec=spec,
        refine_interval=refine_interval,
        summarize_and_reset=summarize_and_reset,
        trace_dir=trace_dir,
        continual=continual,
        continual_lives=continual_lives,
        refiner=refiner,
        refiner_model=refiner_model,
        bootstrap_dir=bootstrap_dir,
        refine=refine,
        setup_tune=tune,
        setup_modify=modify,
        setup_level_blob=level_blob,
        setup_character=character,
        allowed_skill_names=_allowed_skill_names,
        **kwargs,
    )



__all__ = [
    "SYSTEM_PROMPT",
    "_strip_blank_rows",
    "_glyph_run_encode",
    "_inventory_fingerprint",
    "_run_length_encode_messages",
    "_glyph_to_words",
    "_format_obs_balrog",
    "_format_obs_glyphbox",
    "_format_obs_summarize_reset",
    "_descent_status_block",
    "_E1_BEARINGS",
    "_e1_bearing",
    "_e1_classify_frontier",
    "_e1_frontiers_block",
    "_e1_exploration_block",
    "_e1_spatial_belief_block",
    "_VARIANT_FORMATTERS",
    "_paint_frontiers_on_map",
    "format_observation_as_chat",
    "_continual_reset",
    "_write_trace_entry",
    "_drop_before_last_belief",
    "_refinement_directive",
    "_ch_build_window",
    "_ch_inject_system",
    "_ch_save_bootstrap",
    "_compact_chat_history",
    "_STATUS_SIG_RE",
    "_compacted_status_signature",
    "_dedupe_compacted_runs",
    "_msg_role",
    "_msg_content",
    "_replace_content",
    "_one_line_summary",
    "_check_halt_condition",
    "BELIEF_STATE_INTERVAL",
    "_maybe_belief_state_summary",
    "_maybe_distill",
    "_to_action_indices",
    "scout_reward",
    "descent_reward",
    "success_reward",
    "ascension_reward",
    "_ASCENSION_MARKERS",
    "_DEATH_MARKERS",
    "_decode_tty",
    "_detect_terminal_outcome",
    "_code_tool_adapter",
    "_build_skill_adapter_callables",
    "_make_run_macro_adapter",
    "_make_fixed_direction_adapter",
    "_TYPE_MAP",
    "_make_skill_adapter",
    "_iterate_visible_tiles",
    "FRONTIER_STUCK_TURNS",
    "FRONTIER_APPROACH_RADIUS",
    "NEEDS_HIDDEN_TURNS",
    "_update_frontier_blacklist",
    "NetHackVerifiersEnv",
    "load_environment",
    "_build_task_dataset",
    # Prompt factory.
    "ObsSpec",
    "ToolSpec",
    "PromptSpec",
    "build_prompt",
    "VARIANT_REGISTRY",
    "resolve_spec",
    "attach_refiner",
]

