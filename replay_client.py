"""
SonicSight gRPC replay client - stream a video FILE into StreamProcess.

Why this exists
---------------
The live phone test plays the clip through the phone speaker and re-records it
with the phone mic. Measurement of your stream dump showed that round-trip
destroys the signal before the network ever sees it:

    video audio track : flatness 0.183 | 150-600 Hz = 81.0% | 2400+ Hz =  1.0%
    phone mic capture : flatness 0.588 | 150-600 Hz = 11.0% | 2400+ Hz = 43.3%

81% of the music lives in 150-600 Hz. The mic keeps 11-22% of it, and replaces
it with hiss above 2400 Hz. The model cannot separate what it never received.

This client feeds the SAME streaming path (StreamProcess, StreamingBuffer,
overlap-add, jitter-free) with the CLEAN audio and CLEAN frames straight from
the file. No speaker, no mic, no room. It is fully deterministic and repeatable.

What it tells you
-----------------
Run this, then compare to the /predict numbers on the same clip:

  * If replay separation ~= /predict separation, the streaming code is correct
    and the ONLY problem is the acoustic capture loop.
  * If replay is still much worse than /predict, there is a real remaining bug
    in the streaming path, and this harness reproduces it deterministically
    so it can be fixed.

Usage
-----
Put this file in the backend ROOT (next to the `src/` folder), start the server
(python src/run_servers.py), then:

    python replay_client.py --video myclip.mp4

Output lands in ./replay_out/ :
    replay_left.wav     separated left  (what the phone would play)
    replay_right.wav    separated right
    replay_input.wav    the exact clean mixture that was streamed in
    replay_report.txt   metrics, same ones I computed on /predict

Useful flags
------------
    --speed 4           stream 4x faster than realtime (default 1.0)
    --host 1.2.3.4:50051
    --max-seconds 20    only stream the first 20 s
    --no-analyze        skip the metrics section

Requires: grpcio, opencv-python, numpy, ffmpeg on PATH. The backend already
needs all of these, so nothing new to install.
"""

import os
import sys
import time
import wave
import shutil
import argparse
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import grpc
    import sonicsight_pb2
    import sonicsight_pb2_grpc
except Exception:
    print("ERROR: could not import grpc / sonicsight_pb2.")
    print("       Put replay_client.py in the backend ROOT (the folder that")
    print("       contains src/), and make sure grpcio is installed.")
    raise

try:
    import cv2
except Exception:
    print("ERROR: opencv-python (cv2) is required.")
    raise

# These MUST match the server. Imported rather than hardcoded so the harness
# cannot silently drift from config.py.
from config import AUD_RATE, IMG_SIZE, FRAME_RATE

CHUNK_MS = int(round(1000.0 / FRAME_RATE))            # 125 ms at 8 fps
SAMPLES_PER_CHUNK = int(round(AUD_RATE / FRAME_RATE))  # 1378 at 11025/8
JPEG_QUALITY = 90


# ----------------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------------

def extract_audio(video, sr):
    """Decode the video's audio to mono int16 at `sr` using ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH.")
    cmd = [
        "ffmpeg", "-v", "error", "-i", video,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(sr), "-",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg audio extract failed: "
                           + p.stderr.decode("utf-8", "replace")[:400])
    a = np.frombuffer(p.stdout, dtype=np.int16)
    if a.size == 0:
        raise RuntimeError("Video has no decodable audio track.")
    return a.copy()


def extract_frames(video, fps, max_seconds=None):
    """Sample frames at `fps`, split each into left/right halves, resize to
    IMG_SIZE x IMG_SIZE, JPEG encode. Mirrors what the phone sends."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError("Could not open video: %s" % video)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print("  source video : %dx%d  %.2f fps  %d frames" % (w, h, src_fps, total))

    if src_fps <= 0:
        src_fps = 30.0
        print("  WARNING: source fps unknown, assuming 30.")

    step = src_fps / float(fps)
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    out = []
    idx = 0
    next_take = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= next_take:
            next_take += step
            ts_ms = int(round(1000.0 * idx / src_fps))
            if max_seconds is not None and ts_ms > max_seconds * 1000:
                break
            fh, fw = frame.shape[:2]
            mid = fw // 2
            lh = cv2.resize(frame[:, :mid], (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_AREA)
            rh = cv2.resize(frame[:, mid:], (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_AREA)
            okl, lj = cv2.imencode(".jpg", lh, enc)
            okr, rj = cv2.imencode(".jpg", rh, enc)
            if okl and okr:
                out.append((ts_ms, lj.tobytes(), rj.tobytes()))
        idx += 1
    cap.release()

    if not out:
        raise RuntimeError("No frames extracted.")
    print("  sampled      : %d frames at %d fps -> %dx%d halves"
          % (len(out), fps, IMG_SIZE, IMG_SIZE))
    return out


# ----------------------------------------------------------------------------
# streaming
# ----------------------------------------------------------------------------

def build_requests(audio, frames, speed, max_seconds, stats):
    """Generator of StreamChunk. Sends one audio chunk and one frame chunk per
    tick, exactly like the phone does (they are separate messages there too)."""
    n_audio = len(audio) // SAMPLES_PER_CHUNK
    if max_seconds is not None:
        n_audio = min(n_audio, int(max_seconds * FRAME_RATE))

    by_ts = {}
    for ts, lj, rj in frames:
        by_ts[int(ts // CHUNK_MS)] = (lj, rj)

    t0 = time.time()
    for i in range(n_audio):
        ts_ms = i * CHUNK_MS

        if speed > 0:
            target = t0 + (ts_ms / 1000.0) / speed
            dt = target - time.time()
            if dt > 0:
                time.sleep(dt)

        seg = audio[i * SAMPLES_PER_CHUNK:(i + 1) * SAMPLES_PER_CHUNK]
        yield sonicsight_pb2.StreamChunk(
            timestamp_ms=ts_ms,
            audio_pcm=seg.tobytes(),
        )
        stats["audio_sent"] += 1

        fr = by_ts.get(i)
        if fr is not None:
            yield sonicsight_pb2.StreamChunk(
                timestamp_ms=ts_ms,
                left_jpeg=fr[0],
                right_jpeg=fr[1],
                frame_width=IMG_SIZE,
                frame_height=IMG_SIZE,
            )
            stats["frames_sent"] += 1

    yield sonicsight_pb2.StreamChunk(
        timestamp_ms=n_audio * CHUNK_MS,
        is_last=True,
    )
    stats["sent_done"] = True


def write_wav(path, pcm_bytes, sr):
    f = wave.open(path, "wb")
    try:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(sr))
        f.writeframes(pcm_bytes)
    finally:
        f.close()


# ----------------------------------------------------------------------------
# analysis (same metrics I ran on the /predict output)
# ----------------------------------------------------------------------------

def _spec(x, nfft=1024, hop=256):
    w = np.hanning(nfft)
    o = []
    for i in range(0, len(x) - nfft, hop):
        o.append(np.abs(np.fft.rfft(x[i:i + nfft] * w)))
    return np.array(o).T if o else np.zeros((nfft // 2 + 1, 1))


def _flat(x, nfft=1024, hop=256):
    w = np.hanning(nfft)
    fs = []
    for i in range(0, len(x) - nfft, hop):
        s = np.abs(np.fft.rfft(x[i:i + nfft] * w)) + 1e-12
        fs.append(np.exp(np.log(s).mean()) / s.mean())
    return float(np.mean(fs)) if fs else 0.0


def analyze(L, R, MIX, sr, lines):
    def p(s):
        print(s)
        lines.append(s)

    n = min(len(L), len(R))
    if n < 4096:
        p("  not enough output audio to analyze (%d samples)" % n)
        return
    L, R = L[:n], R[:n]

    p("")
    p("=== SEPARATION METRICS (compare to your /predict run) ===")
    p("  reference /predict on this clip:")
    p("      pearson +0.263 | flatness L 0.114 R 0.205 | rms ratio 0.936")
    p("")
    rl = float(np.sqrt((L ** 2).mean()))
    rr = float(np.sqrt((R ** 2).mean()))
    p("  duration        : %.2f s" % (n / float(sr)))
    p("  rms L / R       : %.6f / %.6f   ratio %.3f"
      % (rl, rr, rl / (rr + 1e-12)))
    p("  flatness L / R  : %.4f / %.4f   (1.0 = white noise, <0.25 = tonal)"
      % (_flat(L), _flat(R)))
    pear = float(np.corrcoef(L, R)[0, 1])
    p("  pearson(L,R)    : %+.6f            (lower = better separation)" % pear)

    SL, SR = _spec(L), _spec(R)
    m = min(SL.shape[1], SR.shape[1])
    SL, SR = SL[:, :m], SR[:, :m]
    p("  spec corr       : %+.6f" % float(np.corrcoef(SL.ravel(), SR.ravel())[0, 1]))

    tot = SL + SR + 1e-12
    ml = SL / tot
    p("  implied maskL   : mean %.4f std %.4f" % (ml.mean(), ml.std()))
    p("  frac bins 0.4-0.6: %.4f   (high = 50/50 split = NOT separating)"
      % float(((ml > 0.4) & (ml < 0.6)).mean()))

    p("")
    p("  band split (% of each band's energy going L / R):")
    fr = np.fft.rfftfreq(1024, 1.0 / sr)
    for lo, hi in [(0, 150), (150, 300), (300, 600), (600, 1200),
                   (1200, 2400), (2400, 5512)]:
        k = (fr >= lo) & (fr < hi)
        el = float((SL[k] ** 2).sum())
        er = float((SR[k] ** 2).sum())
        t = el + er + 1e-12
        p("    %5d-%5d Hz   L %6.2f   R %6.2f" % (lo, hi, 100 * el / t, 100 * er / t))

    p("")
    p("=== INPUT SANITY (should look like clean music, NOT like the mic) ===")
    fmix = _flat(MIX)
    b = []
    SM = _spec(MIX)
    tt = float((SM ** 2).sum()) + 1e-12
    for lo, hi in [(150, 600), (2400, 5512)]:
        k = (fr >= lo) & (fr < hi)
        b.append(100 * float((SM[k] ** 2).sum()) / tt)
    p("  streamed mixture: flatness %.4f | 150-600 Hz %.1f%% | 2400+ Hz %.1f%%"
      % (fmix, b[0], b[1]))
    p("  clean reference : flatness 0.183 | 150-600 Hz 81.0% | 2400+ Hz  1.0%")
    p("  phone mic (bad) : flatness 0.588 | 150-600 Hz 11.0% | 2400+ Hz 43.3%")

    p("")
    p("=== VERDICT ===")
    if fmix > 0.40:
        p("  ! Streamed input is already noise-like. Extraction is wrong -")
        p("    this should not happen when reading from the file.")
    elif pear <= 0.35:
        p("  Separation matches the offline ceiling. The streaming CODE IS FINE.")
        p("  The live-app problem is entirely the speaker->room->mic capture.")
    elif pear <= 0.45:
        p("  Close to the offline ceiling but not equal. Minor streaming loss;")
        p("  the dominant problem is still acoustic capture.")
    else:
        p("  Clean input but separation still poor -> a REAL streaming bug")
        p("  remains, and this run reproduces it deterministically. Send me")
        p("  replay_report.txt plus the three wavs.")


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--host", default="localhost:50051")
    ap.add_argument("--out", default=os.path.join(HERE, "replay_out"))
    ap.add_argument("--speed", type=float, default=1.0,
                    help="1.0 = realtime, 4 = 4x faster, 0 = no pacing")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--no-analyze", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print("ERROR: no such video: %s" % args.video)
        return 2
    os.makedirs(args.out, exist_ok=True)

    print("=" * 68)
    print("SonicSight replay client - clean file -> StreamProcess")
    print("=" * 68)
    print("  server       : %s" % args.host)
    print("  AUD_RATE %d | FRAME_RATE %d | chunk %d ms / %d samples"
          % (AUD_RATE, FRAME_RATE, CHUNK_MS, SAMPLES_PER_CHUNK))

    audio = extract_audio(args.video, AUD_RATE)
    print("  audio        : %d samples (%.2f s) peak %d"
          % (len(audio), len(audio) / float(AUD_RATE), int(np.abs(audio).max())))
    frames = extract_frames(args.video, FRAME_RATE, args.max_seconds)

    n_audio = len(audio) // SAMPLES_PER_CHUNK
    if args.max_seconds is not None:
        n_audio = min(n_audio, int(args.max_seconds * FRAME_RATE))
    mix = audio[:n_audio * SAMPLES_PER_CHUNK]
    write_wav(os.path.join(args.out, "replay_input.wav"), mix.tobytes(), AUD_RATE)

    opts = [
        ("grpc.max_send_message_length", 16 * 1024 * 1024),
        ("grpc.max_receive_message_length", 16 * 1024 * 1024),
    ]
    stats = {"audio_sent": 0, "frames_sent": 0, "sent_done": False}
    left, right = bytearray(), bytearray()
    n_res = n_buf = n_err = 0
    hm_min, hm_max = [], []
    t_start = time.time()

    print("")
    print("  streaming (speed %sx) ..." % (args.speed if args.speed > 0 else "max"))
    channel = grpc.insecure_channel(args.host, options=opts)
    try:
        grpc.channel_ready_future(channel).result(timeout=10)
    except Exception:
        print("ERROR: could not reach the gRPC server at %s" % args.host)
        print("       Is `python src/run_servers.py` running?")
        return 3

    stub = sonicsight_pb2_grpc.SonicSightServiceStub(channel)
    try:
        responses = stub.StreamProcess(
            build_requests(audio, frames, args.speed, args.max_seconds, stats))
        for r in responses:
            if not r.success:
                n_err += 1
                if n_err <= 5:
                    print("    server error: %s" % r.error_message)
                continue
            if r.is_buffering:
                n_buf += 1
                continue
            n_res += 1
            if r.left_audio_pcm:
                left.extend(r.left_audio_pcm)
                right.extend(r.right_audio_pcm)
            if r.left_heatmap:
                h = np.frombuffer(r.left_heatmap, dtype=np.uint8)
                if h.size:
                    hm_min.append(float(h.min()) / 255.0)
                    hm_max.append(float(h.max()) / 255.0)
            if n_res % 25 == 0:
                print("    %4d results | %7d samples | t=%5.1fs"
                      % (n_res, len(left) // 2, time.time() - t_start))
    except grpc.RpcError as e:
        print("  RPC ERROR: %s  %s" % (e.code(), e.details()))
    finally:
        channel.close()

    elapsed = time.time() - t_start
    lines = []

    def p(s):
        print(s)
        lines.append(s)

    p("")
    p("=== STREAM SUMMARY ===")
    p("  wall time        : %.1f s" % elapsed)
    p("  audio chunks sent: %d" % stats["audio_sent"])
    p("  frame chunks sent: %d" % stats["frames_sent"])
    p("  results received : %d  (buffering %d, errors %d)" % (n_res, n_buf, n_err))
    p("  samples returned : %d  (%.2f s)" % (len(left) // 2, (len(left) // 2) / float(AUD_RATE)))
    p("  input streamed   : %d  (%.2f s)" % (len(mix), len(mix) / float(AUD_RATE)))
    if len(mix):
        p("  output/input     : %.3f  (1.0 = every input sample came back)"
          % ((len(left) // 2) / float(len(mix))))
    if hm_max:
        p("  heatmap max      : mean %.4f  (1.000 everywhere = saturated)"
          % float(np.mean(hm_max)))
        p("  heatmap min      : mean %.4f" % float(np.mean(hm_min)))

    if len(left) < 4:
        p("")
        p("  NO AUDIO RETURNED. Check the server log.")
        open(os.path.join(args.out, "replay_report.txt"), "w").write("\n".join(lines))
        return 1

    write_wav(os.path.join(args.out, "replay_left.wav"), bytes(left), AUD_RATE)
    write_wav(os.path.join(args.out, "replay_right.wav"), bytes(right), AUD_RATE)

    if not args.no_analyze:
        L = np.frombuffer(bytes(left), np.int16).astype(np.float64) / 32768.0
        R = np.frombuffer(bytes(right), np.int16).astype(np.float64) / 32768.0
        M = mix.astype(np.float64) / 32768.0
        analyze(L, R, M, AUD_RATE, lines)

    with open(os.path.join(args.out, "replay_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("")
    print("  wrote -> %s" % args.out)
    print("     replay_left.wav  replay_right.wav  replay_input.wav  replay_report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
