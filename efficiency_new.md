============================================================
Model        : CATANet (DPRNet)  scale x4
Device       : cuda:0
Mode         : A (fixed LR input)
LR input     : 256 x 256  (measured input)
Eff. output  : 1024 x 1024  (= LR * scale)
------------------------------------------------------------
Params       : 659,707  (659.707 K)
FLOPs        : 52.1370 G   [backend: thop(MACs)]
               (thop reports MACs; multiply by 2 for FLOPs if comparing to FLOPs-based papers)
Latency      : 198.162 ± 9.434 ms  (warmup=20, repeat=100)
============================================================

============================================================
Model        : CATANet (DPRNet)  scale x2
Device       : cuda:0
Mode         : A (fixed LR input)
LR input     : 256 x 256  (measured input)
Eff. output  : 512 x 512  (= LR * scale)
------------------------------------------------------------
Params       : 601,947  (601.947 K)
FLOPs        : 36.1882 G   [backend: thop(MACs)]
               (thop reports MACs; multiply by 2 for FLOPs if comparing to FLOPs-based papers)
Latency      : 183.571 ± 7.700 ms  (warmup=20, repeat=100)
============================================================

============================================================
Model        : CATANet (DPRNet)  scale x3
Device       : cuda:0
Mode         : A (fixed LR input)
LR input     : 256 x 256  (measured input)
Eff. output  : 768 x 768  (= LR * scale)
------------------------------------------------------------
Params       : 674,147  (674.147 K)
FLOPs        : 41.2607 G   [backend: thop(MACs)]
               (thop reports MACs; multiply by 2 for FLOPs if comparing to FLOPs-based papers)
Latency      : 185.880 ± 6.329 ms  (warmup=20, repeat=100)
============================================================