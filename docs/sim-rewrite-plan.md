# Native Simulator Rewrite — Execution Plan

Rebuild the parts of Hollow Knight that the RL agent can observe as a standalone
native simulator. This is not a port of the game. The target is: identical
observation stream, orders of magnitude more throughput, full control.

This document is the contract for the work. Read §2 before writing any code.

---

## 1. Success criterion

One primary number, measured continuously from Phase 2 onward:

**Divergence Horizon (DH)** — given a recorded input sequence and the real game's
recorded state trace, the number of consecutive frames the simulator stays within
tolerance of the trace. Reported over a corpus as p50 / p10 / min.

DH is the gradient. Every phase gate below is expressed in it.

Secondary metrics:

| Metric | Definition |
|---|---|
| Action coverage | implemented `FsmStateAction` types / types present in the dumps |
| Obs parity rate | fraction of replayed steps whose wire bytes match `Net/BinaryProtocol.cs` output exactly |
| Throughput | agent-steps/sec at N worlds |
| Uncited constants | count of magic numbers in `sim/` with no `analysis/` citation. **Must be 0.** |

Bit-exact reproduction across implementations is **not** a goal and is not
achievable — Unity's Box2D reproduces only on the same binary and platform, and
Mono's JIT vs. a native compiler will differ by ~1 ULP per op. Target behavioral
equivalence with explicit tolerances.

---

## 2. Non-negotiable rules

**2.1 Evidence rule.** Every constant, threshold, and behavioral rule in `sim/`
carries a citation to a file in `analysis/`. A value without a citation is a
defect, not a shortcut. This is the single most important rule in the project.

**2.2 No memory-sourced facts.** Do not write Hollow Knight behavior from
recall — not dash speeds, not cooldowns, not frame counts, not "how Unity
usually works." If the decompiled source, a runtime dump, or a recorded trace
does not state it, it is an open question. Log it in
`analysis/open-questions.md` and move on. A plausible-looking wrong constant is
worse than a missing one: it produces a simulator that runs, looks correct, and
teaches the policy timings that fail on the real game.

**2.3 Uncertainty is loud.** Prefer an `UNKNOWN` that aborts over a guess that
runs. Unimplemented `FsmStateAction` types must trap with the type name, not
no-op silently.

**2.4 Harness before implementation.** No subsystem is written before a trace
field exists that measures it. If you cannot measure it, you cannot claim it.

**2.5 Discovery over assumption.** The questions in §5 (P1) are open. Answer
them from evidence. Do not let this document's phrasing imply an answer — where
it describes a possibility, it is describing a branch, not a conclusion.

---

## 3. Repository layout

```
analysis/              evidence. verbose by design. never imported by sim/.
  decomp/              ilspycmd output of Assembly-CSharp (gitignore the tree;
                       cite paths + line numbers regardless)
  dumps/               runtime reflection dumps: hero fields, physics settings,
                       layer matrix, time settings
  fsm/                 FsmDumper output, one JSON per scene
  traces/              recorded (input, state) pairs captured from real HK
  specs/               subagent findings, one .md per subsystem, citation-dense
  open-questions.md    every unresolved behavioral question, with owner
sim/                   the rewrite. terse, cited, no prose.
  core/                world state, fixed-step loop, allocation-free
  hero/
  fsm/                 interpreter + one file per action type
  phys/
  obs/                 wire encoder, byte-identical to Net/BinaryProtocol.cs
harness/               differential runner, divergence reporter, corpus tools
docs/                  cross-cutting policy ONLY (see §7). three files, not thirty.
```

The `analysis/` ↔ `sim/` split is load-bearing: evidence is allowed to be
verbose, code is required to be terse, and code points at evidence.

---

## 4. Language and build

Choose for: no GC pauses, C ABI for Python FFI, explicit float control,
struct-of-arrays layout, vectorizable across N worlds. C, Rust, or Zig all
qualify. Pick one in Phase 0 and record the decision.

Float determinism is a build contract, not an afterthought. Disable fast-math
and FP contraction (`-ffp-contract=off` or the equivalent), pin the FP
environment, and document the choice in `docs/float-parity.md` before any
physics code lands. Changing these flags later invalidates every DH measurement
taken before the change.

---

## 5. Phases

Each phase has an entry condition, work, and a **gate** that is a measurement,
not a judgement. Do not advance on a gate you have not run.

### P0 — Build the oracle

The mod becomes a recorder before anything is rewritten. Nothing downstream is
trustworthy without this.

- Extend the C# mod with a per-frame full-state trace writer: binary, versioned
  schema, covering every field the simulator will have to reproduce. Capture at
  the finest granularity the mod can reach, not at agent-step granularity.
- Add runtime reflection dumpers. `Game/FsmDumper.cs:227` already contains a
  generic `GetFields(Public | Instance)` walker — point it at
  `HeroController.instance`, `PlayerData.instance`, the hero's `Rigidbody2D` and
  colliders, `Physics2D` settings, the layer collision matrix, and `Time`
  settings. Runtime values are ground truth; a decompiled field initializer is
  not, because Unity serializes inspector overrides into the scene/prefab asset.
- **Run `FsmDumper` for the first time.** It has never executed —
  `python/fsm_dumps/` is gitignored and `state_graphs/` is empty. Every claim
  about "the boss is solved as data" rests on unvalidated code. Validate the
  output before building on it.
- Decompile `Assembly-CSharp.dll` with `ilspycmd` into `analysis/decomp/`.

**Gate — the most important experiment in the project.** Feed a recorded input
sequence back into the real game from a cold scene load and record a second
trace. Do the two traces match, and for how long?

This measures whether Hollow Knight can reproduce *itself*. Unity's own guidance
is that 2D physics is deterministic on the same machine only after a full scene
reload that recreates the Box2D world. If the real game is not self-reproducible
under your harness conditions, DH against a recorded trace is meaningless and
the entire verification strategy needs redesigning — **you must know this before
writing a line of simulator.** Record the observed self-DH; it is the ceiling on
every number that follows.

### P1 — Discovery fan-out (read-only, parallel, no code)

Subagents produce specs in `analysis/specs/`. Every claim carries a
`file:line` citation. Every unanswered question goes to `open-questions.md`.

Questions, roughly in value order:

1. **Is hero motion solver-driven or velocity-driven?** Does `HeroController`
   assign `rb2d.velocity` directly and detect collisions with casts, does it
   apply forces and let Box2D's sequential-impulse solver resolve contacts, or
   is it a mix that varies by state (grounded / airborne / dashing / wall)?
   This determines whether the project needs a physics solver at all, and it is
   the largest single branch in the plan. Answer it with a runtime experiment
   (compare the velocity HeroController assigns against the velocity present
   after the physics step), not by reading alone.
2. **Frame ordering.** Where do `HeroController`, PlayMaker FSMs, `tk2d`
   animators, and the mod's step coroutine sit relative to each other and to
   `FixedUpdate`? Unity's documented order has `FixedUpdate` running zero, one,
   or many times per frame and `yield return null` resuming after all `Update`
   calls on the next frame — but script execution order settings and the
   `Time.captureDeltaTime` + frame-skip regime this project runs under
   (`Environment/TrainingEnv.cs:119`, `kStepDeltaTime = 0.00848f`) can change
   what that means in practice. Determine the actual observed order and the
   fixed-timestep accumulator's behaviour under capture mode. Off-by-one-frame
   divergence lives here.
3. **The nine `Can*` predicates** (`Game/StateExtractor.cs:50-58`) — exact
   conjunctions over hero state, since these are directly observed by the policy.
4. **Damage path** — `HealthManager`, i-frame windows, recoil, hit-stop, and how
   `is_invincible` becomes true.
5. **tk2d animator semantics** — frame advance, wrap modes, and how
   `CurrentFrame` relates to elapsed time.
6. **FsmStateAction census** — which types actually appear across the dumps,
   ranked by frequency. This is the implementation backlog, ordered.

**Gate:** every spec claim cited; a separate citation-checking agent has
re-read each cited location and confirmed it says what the spec claims;
question 1 answered by experiment.

### P2 — Harness

- Differential runner: load a trace, drive the sim with the recorded inputs,
  compare per frame, report the **first** divergence — frame index, field name,
  expected, actual, magnitude. First-divergence reporting is the entire debugging
  loop; a summary of total mismatches is useless by comparison.
- Corpus tooling: named input sequences, tagged by what they exercise.

**Gate:** a null model — a "simulator" that simply echoes the recorded trace —
scores DH = full length. Perturbing any single field by one ULP drops DH to that
frame. If the harness cannot detect a one-ULP perturbation it is not measuring
what you think it is.

### P3 — Hero in an empty room

Terrain only. No boss, no combat, no damage.

**Gate:** DH on a movement corpus (walk, jump, double jump, wall interaction,
dash, fall) at a target chosen from P0's measured self-DH ceiling.

### P4 — FSM runtime + one boss

- Interpreter over the `analysis/fsm/` dumps.
- Action types implemented in census order from P1.6. Unimplemented types trap.
- Start with a **boss-only trace** (agent idle) so boss determinism is isolated
  from hero coupling.

**Gate:** boss active-state sequence matches the recorded trace for a full
episode, including branch outcomes under a seeded RNG.

### P5 — Combat coupling

Damage in both directions, hitbox spawn/despawn timing, i-frames, recoil.

**Gate:** DH on full-fight traces; damage events match frame-exactly.

### P6 — Observation parity

The simulator emits `Net/BinaryProtocol.cs` bytes exactly and speaks the same
WebSocket protocol, so the existing Python trainer connects **unmodified**. This
is deliberately the slow path — it exists to prove parity, not to be fast.

**Gate:** byte-identical step payloads for a replayed trace, verified against
`python/binary_protocol.py`.

### P7 — Speed

Only now: struct-of-arrays, N worlds, vectorization, in-process FFI replacing
the WebSocket.

**Gate:** throughput target met **and DH unchanged**. An optimization that moves
DH is a behavior change and must be reverted or re-justified.

### P8 — Transfer

Train in sim, evaluate in real HK. Domain-randomize every constant flagged
uncertain in `open-questions.md` — this converts unresolved fidelity risk into
policy robustness, and changes the requirement from "exact" to "unbiased with
known spread."

**Gate:** sim-trained policy's real-HK performance against a real-HK-trained
baseline.

---

## 6. Agent orchestration

Orchestrator: Fable. Workers: Opus subagents.

### Orchestrator rules

- **Never read `analysis/decomp/` directly.** It will flood context and force
  compaction. Delegate all decompiled-source reading; retain only specs.
- **Externalize state before context fills.** The plan, the phase, the current
  DH, and the open-questions list live in files, not in conversation. They must
  survive compaction.
- **Own the shared contracts.** Trace schema, world-state struct, interpreter
  ABI, action dispatch registry. Workers never edit these.

### What parallelizes

Fan out 4–8 concurrent workers for genuinely independent, self-contained work:

- **P1 discovery** — read-only, disjoint subsystems, no shared state. This is
  the best fan-out in the project.
- **P4/P5 action-type implementations** — after the interpreter contract is
  frozen. One file per action type, disjoint ownership.
- **Corpus generation** — independent input sequences.

### What does not parallelize

- Simulator core and data layout — one owner, sequential.
- **Divergence debugging** — inherently serial. Fix the first divergence, re-run,
  find the next. Parallel agents chasing divergences at different frames will
  produce conflicting fixes for the same root cause.
- Anything before the contracts it depends on exist.

Multi-agent orchestration costs roughly 15× the tokens of a single thread and
only wins when work decomposes into independent parallel threads. Fanning out
serial work spends the tokens and gets none of the benefit.

### Anti-conflict protocol

Parallel writers require **disjoint file ownership**. Shared registries (action
dispatch table, field lists, schema definitions) are orchestrator-only: a worker
adds its own file and reports the exported symbol; the orchestrator wires it up.
Use git worktrees only when disjoint ownership cannot be arranged.

### Verification passes

Run a separate agent — never the author — for:

- **Citation checking.** Re-read every cited `file:line` and confirm it says what
  the spec claims. This catches the project's dominant failure mode (confident
  fabrication) and is cheap.
- **Uncited-constant sweep.** Grep `sim/` for numeric literals without an
  adjacent citation comment. Must return zero.
- **Adversarial review** before any phase gate is declared passed.

---

## 7. Documentation policy

**Mandatory:**
- A citation comment on every constant and every non-obvious behavioral rule:
  `// analysis/decomp/HeroController.cs:1841`
- Exactly three cross-cutting policy documents, because these span every file and
  cannot live next to any single one:
  - `docs/float-parity.md` — FP flags, tolerance definitions, what "match" means
  - `docs/frame-order.md` — the execution order the sim implements, and why
  - `docs/trace-format.md` — schema, versioning, capture conditions

**Forbidden:**
- Prose restating what the code already says
- File headers, per-function docblocks on obvious functions, section banners
- Any fourth document in `docs/` without a stated reason it cannot be a comment

`analysis/specs/` is evidence, not documentation. It is allowed to be long, and
it is the only place extended prose belongs.

---

## 8. Open questions — do not assume answers

Seed `analysis/open-questions.md` with at minimum:

- Is the real game self-reproducible under trace replay, and for how many frames?
- Is hero motion solver-driven, velocity-driven, or state-dependent?
- Does `Time.captureDeltaTime` alter the `FixedUpdate` accumulator, and does it
  override `Time.unscaledDeltaTime`? (This second part also determines whether
  the existing `perf | sim_pct` metric in `python/train.py:985` is measuring
  anything real — the tell is `sim_pct ≥ 100%` with `overhead 0.0ms`.)
- Which `FsmStateAction` types appear in practice, and what does each do?
- Are boss branch decisions seeded from a source the simulator can reproduce?
- What tolerance constitutes "matching" for each traced field?

Every one of these has a discoverable answer on the machine running the game.
None of them should be answered from recall.

---

## 9. First week

1. Decompile `Assembly-CSharp.dll`; commit nothing but the paths you cite.
2. Run `FsmDumper` and validate its output against the decompiled FSM data.
3. Write the reflection dumpers; capture `analysis/dumps/`.
4. Write the trace recorder; capture one 600-frame trace.
5. **Run the P0 self-reproducibility gate.** Report the number.

Step 5 decides the shape of everything after it. Do not skip ahead of it.
