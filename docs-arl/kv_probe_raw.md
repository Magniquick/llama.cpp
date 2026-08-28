# Local probe: KV precision vs context (ornith-35b-armB-ABCD, u4 dense, no draft, perf/80W)

Columns are the probe's decode-only t/s. Context is L/2, the mean over a generation of L
tokens from a short prompt — exact for a linear time-vs-context law.

| generated | mean ctx | f16 KV | u8 KV |
|-----------|----------|--------|-------|
| 2048      | 1024     | 26.64  | 27.04 |
| 8192      | 4096     | 25.29  | 27.40 |
| 16384     | 8192     | 22.67  | 26.93 |

f16 fit: 36.28 ms + 0.9278 ms per 1k context tokens -> implied KV-path bandwidth 22.1 GB/s,
against 84.6 GB/s for weight streaming. Crosses the 20 t/s floor at ~14.8k context.

u8 is essentially FLAT to 16k (+18.8% at 16k). This REVERSES the note in ov_serve.py
("f16 ... faster here (29.9 vs 26.6 t/s)"), which was measured at short context where the
int8 quantisation overhead dominates and the byte saving has not yet paid for itself.

Consequence: the throughput floor is NOT context-limited once KV is u8, so the 32k
scenario that appeared to break every candidate does not.
