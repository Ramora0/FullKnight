# Where to look

Vague pointers, not a checklist. Read `session.md` for current priority.
The point of this file is to suggest *directions* — figure out the
specifics yourself by reading the code and the HK decompiled source.

## Probably free (no behavior risk)

- **Audio**: HK is loud. Listeners, sources, mixers, music coroutines.
- **Visuals the agent can't see**: particles, post-processing, UI canvases,
  lighting, screen shake, dialogue.
- **Ambient world MonoBehaviours**: idle background NPCs, scene-decoration
  scripts, room-scale effects.

## Behavior-adjacent (verify quality CIs hold)

- **Lower the per-game-second frame rate via dt + frames_per_wait**: with
  captureDeltaTime=0.00283 and timeScale=3, each Unity frame represents
  0.00848 game-sec → **118 game-fps**. That's way more physics/animation
  resolution than HK actually needs. Increase captureDeltaTime (and
  inversely decrease frames_per_wait) so the agent's gtime stays at
  baseline (0.0424s/step) but each agent step traverses fewer Unity
  frames. e.g. captureDeltaTime=0.0141 + frames_per_wait=1 → 71 game-fps,
  1 frame per agent decision. Going as low as 24 game-fps is plausible
  before HK FSM/animator timing starts degrading. Less compute per agent
  step → less wallclock per env-step. Quality risk: animation events and
  collision tickets may fire at different fractions of game-time, similar
  flavor of regime risk to dt-pinning.
- **Physics rate**: `Time.fixedDeltaTime` may be tuned for 60Hz wallclock
  but we're running at `time_scale=3`.
- **Animator culling / disabling**: animators drive attack telegraphs via
  events — be careful, but most off-screen ones are dead weight.
- **Collision matrix**: lots of layer pairs probably collide that don't
  matter for combat.
- **Hitbox observation path**: per-collider `HitboxReader` MonoBehaviours
  in `Environment/HitboxObserver.cs`. Allocations and Update calls scale
  with collider count.

## GC pressure

Watch `gc_heap_growth_mb` and `rtime_p95_ms`. If either is high, the
bottleneck is probably allocations, not compute. Look at:

- The per-step path in `TrainingEnv.cs` and `BinaryProtocol.cs`.
- LINQ, `new List<T>()`, string interpolation, transform.Find by name.
- Unity APIs that return new arrays (`FindObjectsOfType`, etc.).

## Structural (last-resort, big)

- Single-manager hitbox observer instead of per-collider MB.
- Custom combat-only mini-scene that skips most of HK's lifecycle.
- Hand-rolled AABB sweep instead of `Physics2D.Simulate`, if physics is
  the dominant cost.

## When stuck

Re-read `Environment/TrainingEnv.cs` and `HitboxObserver.cs` for new
angles. Scan the HK decompiled source at
`C:\Users\Lee\coding\CSharp\HK\decomp\assembly-csharp\` for
MonoBehaviours that aren't combat-relevant. Watch the `run.log` for
log spam and exception traces — both are real allocation cost.
