# SonicSight multi-model integration — test plan

**Status: the integration is UNVALIDATED.** The multisensory model itself is
thoroughly validated (latency 142.6 ms median, CAM cross-check margin +0.8248
PASS, CPU/GPU equivalence to 1 ULP — see the project handoff). Everything built
on top of it — the engine adapter, the proto changes, the registry refactor,
and the mobile toggle — has been proven only at the compile/import/unit-test
level, in an environment with no checkpoints, no TF runtime, and no device.
Nothing below should be described as working until its gate has actually run.

What already ran (this workspace): 36 backend unit tests (registry constants,
buffer instance-state isolation, frame selection equivalence, CAM postprocess
semantics, resampler properties, metadata branch), proto stub wire-compat
checks, `assembleDebug` for every mobile phase.

Environment for everything below: the WSL2 GPU machine with both checkpoint
sets, per the handoff's environment notes.

---

## T1 — Sound of Pixels byte-identical gate (the phase 2 promise)

Proves the registry refactor changed nothing numerically.

1. Check out backend `main` (pre-refactor). Start the server fresh.
   `python replay_client.py --video <fixed_clip>.mp4 --max-seconds 20`; save
   `replay_out/` as `baseline_A`. Restart the server, repeat → `baseline_B`.
2. `cmp baseline_A/replay_left.wav baseline_B/replay_left.wav` (and right).
   - Identical → the pipeline is run-deterministic; the gate is byte equality.
   - Different → `cudnn.benchmark`/AMP nondeterminism; the gate becomes
     max-abs-sample-delta between refactor and baseline ≤ the baseline's own
     run-to-run delta. Record which case you're in.
3. Check out `phase3-multisensory` (contains phase 2). Fresh server, same clip,
   same flags → `refactored`. Compare against baseline per step 2's rule.

**Pass:** audio and heatmap outputs equal under the applicable rule.
**Fail action:** diff `StreamingBuffer` behaviour first (window starts, frame
picks) via `SONICSIGHT_DUMP_STREAM=1` cycle logs before suspecting anything else.

## T2 — Multisensory engine vs the validated probe

Proves the adapter reproduces the measurement record on the same input.

1. `sep_cam_probe.py --vid_file <clip> --start S --save_npy --warm 10` →
   probe `cam.npy`, fg/bg wavs, latency numbers.
2. Feed the same window through `MultisensoryEngine.eval_stream_window`
   (small harness: load the same 2.135 s of audio/frames the probe used).
3. Compare: raw CAM stats (min/max/std, positive fraction) must match the
   probe to float precision; the engine's reduced+normalized map must equal
   `cam_analyze.reduce_positive` + max-norm of the probe's CAM (the adapter
   deliberately reproduces it; any drift is a bug). fg/bg audio should match
   the probe's after accounting for the adapter's input-gain undo and
   21000→22050 resample.
4. Latency: median over ≥10 calls ≤ ~150 ms; confirm 250 ms hop holds with
   headroom.

**Pass:** probe-equal CAM numbers, latency within the validated envelope.

## T3 — VRAM with BOTH models resident (handoff open risk #3)

Never measured: the handoff's 1870 MiB TF figure was taken without PyTorch
loaded. Protocol:

1. Fresh boot / no other GPU processes. Record `nvidia-smi` idle.
2. Start `run_servers.py` (loads PyTorch engine + TF engine via
   `load_all_engines`). Record after load.
3. Run a 60 s sonicsight stream, then a 60 s multisensory stream, then one of
   each again (exercises both allocators' steady state). Record after each.

| checkpoint | used MiB | free MiB |
|---|---|---|
| idle | TBD | TBD |
| both models loaded | TBD | TBD |
| after sonicsight stream | TBD | TBD |
| after multisensory stream | TBD | TBD |
| steady state (both exercised) | TBD | TBD |

**Pass:** steady state fits 6 GB with ≥ 500 MiB free. If it doesn't:
`MS_GPU_MEM_FRACTION` is the knife — cap TF and re-measure latency (the 512 MB
cuDNN cap already bought ~3 GB; do not touch it).

## T4 — gRPC contract integration

1. `HealthCheck` → `loaded_models` lists exactly what loaded (try all three
   states: both, one, none).
2. Metadata `sonicsight-model: multisensory` on a stream → results echo
   `model_id=multisensory`, `heatmap_count=1`, heatmap bytes are 3136 (56×56)
   or empty when gated, `cam_confidence` populated.
3. Unknown model id → `FAILED_PRECONDITION` naming the available ids;
   known-but-unloaded → `FAILED_PRECONDITION` "not loaded".
4. Old-client compatibility: a stream with NO metadata behaves exactly as
   pre-registry (sonicsight, results parse with new fields defaulted).
5. Note: `replay_client.py` speaks the sonicsight capture profile only. For a
   deterministic multisensory replay it needs a `--model multisensory` mode
   (full_jpeg letterbox + 22050 Hz audio + metadata). Until that exists, T5's
   live phone test is the multisensory end-to-end.

## T5 — Mobile end-to-end (the phase 4 gate)

On a device, against the WSL2 server (portproxy/mirrored networking):

1. Music model: point at an instrument video/speaker setup; heatmap follows the
   sound source; Left/Right solo chips actually solo; audio sounds separated.
2. Switch to Speech mid-session: stream cancels, reopens; no stale overlay or
   audio from the old model (drop-filter working); labels flip to
   On-screen/Off-screen, legend flips to Matches audio/No match.
3. Speech model on a talking person: heatmap sits on the speaker; On-screen
   solo isolates their voice by ear; playback rate correct (no chipmunk/slow
   audio — that is the 22050 vs 11025 wiring).
4. Switch back. Repeat 3×; watch for leaks (`adb shell dumpsys meminfo`).
5. **Record the encode numbers** from the `SonicSightPerf` 5 s summaries on the
   speech model: frames sent/s vs the 30/s target, encode avg/max ms. This is
   the measured answer to the "30 JPEG encodes/s" risk — write the numbers
   down, per device tested.
6. First-result latency by stopwatch vs the buffering copy (~1.2 s speech,
   ~3.0 s music perceived).

## T6 — Honest failure states

1. Server down → "Can't reach the server at <host>…" with settings path; no
   crash; recoverable after server returns.
2. Multisensory checkpoint removed → HealthCheck omits it; selecting Speech
   streams `FAILED_PRECONDITION` → "Speech isn't loaded on this server…".
3. Low confidence: speech model pointed at a wall while audio plays off-screen
   → gate pill appears ("no confident on-screen source"), overlay clears,
   audio keeps playing; recovery when a speaker enters frame.
4. Reduced motion (developer options → animator duration 0): no fades, static
   buffer hairline; TalkBack: every control announced, state changes spoken
   (live regions).

---

Order: T1 → T2 → T3 → T4 → T5 → T6. Stop at the first failure; later tests
assume earlier ones passed.
