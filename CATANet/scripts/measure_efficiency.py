"""效率测量脚本：统计 DPRNet / CATANet 的 Params、FLOPs、推理时延。

统一口径（论文 Table II 必须在正文说明）：
- Params：模型全部可训练 + 不可训练参数总量。
- FLOPs：按固定 **输出** 尺寸（默认 1280x720）反推 LR 输入尺寸（output / scale），
  与轻量 SR 社区惯例一致，便于跨尺度、跨方法公平对比。
- 推理时延：单 GPU，固定 LR 输入尺寸，warmup 后多次前向取均值±标准差（ms）。

用法（在装好 torch 的训练机执行）：
    python scripts/measure_efficiency.py --scale 2
    python scripts/measure_efficiency.py --scale 3 --device cuda:0
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
    parser.add_argument('--output-h', type=int, default=720, help='fixed HR output height for FLOPs (default 720)')
    parser.add_argument('--output-w', type=int, default=1280, help='fixed HR output width for FLOPs (default 1280)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--warmup', type=int, default=20, help='warmup iterations before timing')
    parser.add_argument('--repeat', type=int, default=100, help='timed forward passes for latency averaging')
    args = parser.parse_args()

    device = torch.device(args.device)

    # 由固定输出尺寸反推 LR 输入尺寸；不整除时对 LR 向下取整（与轻量 SR 社区惯例一致，
    # 例如 x3 下 1280/3 -> LR 宽 426，有效输出 1278x720，误差可忽略）
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
    print(f'HR output    : {args.output_h} x {args.output_w} (requested for FLOPs)')
    print(f'Eff. output  : {eff_h} x {eff_w}  (= LR * scale, 不整除时向下取整)')
    print(f'LR input     : {lr_h} x {lr_w}  (= output // scale)')
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
