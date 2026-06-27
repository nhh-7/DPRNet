"""效率测量脚本：统计 DPRNet / CATANet 的 Params、FLOPs、推理时延。

支持两种口径（论文 Table II 必须在正文说明用的是哪一种，且全表统一）：
- 【A 固定 LR 输入】--lr-h/--lr-w（推荐，对齐 CATANet 论文 Table 6 的 256x256 输入）：
  直接给定 LR 输入尺寸，FLOPs/时延都在该输入上测；与 scale 无关地固定输入面积，
  是 CATANet/SwinIR 等论文报 Multi-Adds 的口径（input 256x256 -> x4 输出 1024x1024）。
- 【B 固定 HR 输出】--output-h/--output-w（旧默认 720x1280）：按输出反推 LR 输入（output/scale）。
  ⚠ 与 CATANet 论文 46.8G(256x256 输入) 不可直接比，仅用于我方三尺度自比。

Params：模型全部可训练 + 不可训练参数总量。
推理时延：单 GPU，固定 LR 输入尺寸，warmup 后多次前向取均值±标准差（ms）。

用法（在装好 torch 的训练机执行）：
    # 与 CATANet 论文同口径重测（推荐，对齐 46.8G）：
    python scripts/measure_efficiency.py --scale 4 --lr-h 256 --lr-w 256
    python scripts/measure_efficiency.py --scale 2 --lr-h 256 --lr-w 256
    # 旧口径（固定 HR 输出反推 LR，仅供我方三尺度自比）：
    python scripts/measure_efficiency.py --scale 4 --output-h 720 --output-w 1280

FLOPs 后端优先 thop，回退 fvcore；两者都没有时仅报 Params + 时延并提示安装。
"""
import argparse
import statistics
import time

import torch

from basicsr.archs.catanet_arch import CATANet


def build_model(scale: int) -> torch.nn.Module:
    """构建与训练完全一致的默认 CATANet（DPRNet）结构。"""
    model = CATANet(upscale=scale)
    model.eval()
    return model


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_flops(model: torch.nn.Module, lr_input: torch.Tensor):
    """返回 (flops, backend_name)；无可用后端时返回 (None, None)。"""
    # 优先 thop
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(lr_input,), verbose=False)
        return float(flops), 'thop(MACs)'
    except ImportError:
        pass
    # 回退 fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, lr_input).total()
        return float(flops), 'fvcore(FLOPs)'
    except ImportError:
        return None, None


@torch.no_grad()
def measure_latency(model: torch.nn.Module, lr_input: torch.Tensor,
                    warmup: int, repeat: int, device: torch.device):
    is_cuda = device.type == 'cuda'
    # warmup
    for _ in range(warmup):
        model(lr_input)
    if is_cuda:
        torch.cuda.synchronize()

    timings = []
    for _ in range(repeat):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(lr_input)
        if is_cuda:
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)  # ms

    mean = statistics.mean(timings)
    std = statistics.pstdev(timings) if len(timings) > 1 else 0.0
    return mean, std


def human_params(n: int) -> str:
    if n >= 1e6:
        return f'{n / 1e6:.3f} M'
    return f'{n / 1e3:.3f} K'


def main():
    parser = argparse.ArgumentParser(description='Measure Params / FLOPs / latency for CATANet (DPRNet).')
    parser.add_argument('--scale', type=int, required=True, choices=[2, 3, 4],
                        help='super-resolution scale factor')
    # 口径 A（推荐）：固定 LR 输入尺寸，直接对齐 CATANet 论文 256x256 输入
    parser.add_argument('--lr-h', type=int, default=None,
                        help='[mode A] fixed LR input height; if set, overrides --output-* (CATANet paper uses 256)')
    parser.add_argument('--lr-w', type=int, default=None,
                        help='[mode A] fixed LR input width; if set, overrides --output-* (CATANet paper uses 256)')
    # 口径 B（旧）：固定 HR 输出尺寸反推 LR
    parser.add_argument('--output-h', type=int, default=720, help='[mode B] fixed HR output height (default 720)')
    parser.add_argument('--output-w', type=int, default=1280, help='[mode B] fixed HR output width (default 1280)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--warmup', type=int, default=20, help='warmup iterations before timing')
    parser.add_argument('--repeat', type=int, default=100, help='timed forward passes for latency averaging')
    args = parser.parse_args()

    device = torch.device(args.device)

    # 口径 A：固定 LR 输入（推荐，对齐 CATANet 论文）；只要给了 --lr-h/--lr-w 之一即启用
    if args.lr_h is not None or args.lr_w is not None:
        if args.lr_h is None or args.lr_w is None:
            parser.error('--lr-h and --lr-w must be provided together (mode A: fixed LR input).')
        mode = 'A (fixed LR input)'
        lr_h, lr_w = args.lr_h, args.lr_w
        eff_h, eff_w = lr_h * args.scale, lr_w * args.scale
    else:
        # 口径 B：由固定输出尺寸反推 LR 输入；不整除时对 LR 向下取整（与轻量 SR 社区惯例一致，
        # 例如 x3 下 1280/3 -> LR 宽 426，有效输出 1278x720，误差可忽略）
        mode = 'B (fixed HR output)'
        lr_h = args.output_h // args.scale
        lr_w = args.output_w // args.scale
        eff_h = lr_h * args.scale
        eff_w = lr_w * args.scale

    model = build_model(args.scale).to(device)
    lr_input = torch.randn(1, 3, lr_h, lr_w, device=device)

    params = count_params(model)
    flops, backend = measure_flops(model, lr_input)
    mean_ms, std_ms = measure_latency(model, lr_input, args.warmup, args.repeat, device)

    print('=' * 60)
    print(f'Model        : CATANet (DPRNet)  scale x{args.scale}')
    print(f'Device       : {device}')
    print(f'Mode         : {mode}')
    print(f'LR input     : {lr_h} x {lr_w}  (measured input)')
    print(f'Eff. output  : {eff_h} x {eff_w}  (= LR * scale)')
    print('-' * 60)
    print(f'Params       : {params:,}  ({human_params(params)})')
    if flops is not None:
        print(f'FLOPs        : {flops / 1e9:.4f} G   [backend: {backend}]')
        print('               (thop reports MACs; multiply by 2 for FLOPs if comparing to FLOPs-based papers)')
    else:
        print('FLOPs        : N/A  (install thop or fvcore: `pip install thop`)')
    print(f'Latency      : {mean_ms:.3f} ± {std_ms:.3f} ms  '
          f'(warmup={args.warmup}, repeat={args.repeat})')
    print('=' * 60)


if __name__ == '__main__':
    main()
