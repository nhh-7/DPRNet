#!/usr/bin/env python3
"""Smoke test for the EMA-center routing arm (C1 comparison). Run on the training box.

Verifies, WITHOUT any training, that:
  1. Both routing modes build: CATANet(routing_mode='dpr') and ('ema_center').
  2. Both run a forward pass at x2/x3/x4 and return the correct SR output shape.
  3. EMACenterRouter exposes the interface the model relies on (aux_loss, last_usage,
     router_logit_scale, max_router_logit_scale) so loss aggregation + diagnostics work.
  4. The EMA centers actually update across training steps (buffer changes in train mode).
  5. Parameter counts are reported for both arms (sanity for the C1 fairness discussion).

Usage (from CATANet/, on a machine with torch installed):
    python scripts/smoke_test_ema_router.py
Exit code 0 = all checks passed.
"""
import sys
import torch

from basicsr.archs.catanet_arch import CATANet, EMACenterRouter, DPR


def _forward_shape_ok(mode, scale):
    net = CATANet(upscale=scale, routing_mode=mode)
    net.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = net(x)
    exp = (1, 3, 32 * scale, 32 * scale)
    assert tuple(y.shape) == exp, f"{mode} x{scale}: got {tuple(y.shape)}, want {exp}"
    return net


def main():
    torch.manual_seed(0)

    # 1-2. both modes build + forward at every scale
    for mode in ("dpr", "ema_center"):
        for scale in (2, 3, 4):
            _forward_shape_ok(mode, scale)
            print(f"[ok] {mode:11s} x{scale} forward shape")

    # 3. interface parity: the model reads these off tab.dpr / the router module
    net = CATANet(upscale=4, routing_mode="ema_center")
    tab0 = net.blocks[0][0]
    router = tab0.dpr
    assert isinstance(router, EMACenterRouter)
    for attr in ("aux_loss", "last_usage", "router_logit_scale", "max_router_logit_scale"):
        assert hasattr(router, attr), f"EMACenterRouter missing {attr}"
    assert hasattr(tab0, "num_tokens")
    print("[ok] EMACenterRouter interface parity (aux_loss/last_usage/router_logit_scale)")

    # 4. EMA centers update in train mode
    net.train()
    before = router.means.clone()
    x = torch.randn(2, 3, 32, 32)
    _ = net(x)
    after = router.means
    delta = (after - before).abs().sum().item()
    assert delta > 0, "EMA centers did not update during a train-mode forward"
    print(f"[ok] EMA centers update in train mode (L1 delta={delta:.4f})")

    # aux_loss must stay None for EMA routing (no balance loss added to l_total)
    assert router.aux_loss is None, "EMA arm should not produce a balance aux_loss"
    print("[ok] EMA arm produces no balance aux_loss")

    # 5. param counts for the fairness note
    def nparams(m):
        return sum(p.numel() for p in m.parameters())
    p_dpr = nparams(CATANet(upscale=4, routing_mode="dpr"))
    p_ema = nparams(CATANet(upscale=4, routing_mode="ema_center"))
    print(f"[info] x4 params: DPR={p_dpr/1e3:.2f}K  EMA-center={p_ema/1e3:.2f}K")

    print("\nALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
