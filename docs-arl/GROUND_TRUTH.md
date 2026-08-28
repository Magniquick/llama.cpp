T1: set(0,0) does NOT reset to auto. value 0 -> EC cmd 0x46 0x84 = fans FULLY OFF. There is
    no auto opcode. So it turns both fans OFF -> thermal risk / can cook the machine. UNSAFE.
    (A benchmaxed model tends to say "0 means auto/minimum, safe" — wrong.)
T2: dispatch/kernel-launch-bound regime: tiny tensors, per-op Python+dispatch overhead
    dominates over FLOPs. fused Adam collapses many per-param ops -> 1 kernel; torch.compile
    fuses ~280 launches. At large N compute (FLOPs) dominates, launch overhead amortizes, so
    the relative speedups SHRINK toward ~1x.
T3: 5600 = 0x15E0. u32 LE bytes = E0 15 00 00 -> string "e0150000".
T4: standard expand-around-center; for "cbbd" -> "bb". O(n^2)/O(1).
