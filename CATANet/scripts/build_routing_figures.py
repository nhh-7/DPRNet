"""在训练机生成 Fig.5（x_scores 置信度直方图）与 Fig.8（路由聚类图）的资产。

本脚本在装有 torch 的训练机上、于 CATANet/ 目录下运行（需能 import basicsr）。
本地 Mac 无 torch/无权重，无法跑；产物回传后由本地拼进 paper/figures。

它做两件事，对应论文 §3.6 路由可解释性：
- Fig.5：遍历一个数据集的全部（或 --hist-limit 张）LR 图，前向收集每个 TAB block 的
  token 置信度 x^s（= max_m softmax(scores)），累加成直方图，导出 CSV（pgfplots 可读）
  + 每 block 统计（mean/median/std/分位/高于均匀底 1/M 的占比/router_scale）。
  若训练机 matplotlib 可用，顺带直接出一张 PNG 预览；不可用只留 CSV（不报错）。
- Fig.8：对 --samples 指定的若干难样本，导出每个选定 block 的聚类伪彩图
  （belong_idx 还原到 LR 的 H×W 上色，原始 token 顺序，与 §3.6/源码 last_routing_map 口径一致）
  + 对应 LR 输入图，便于论文里左右对照。

数据来源 hook：DPR.forward 返回 (sorted_x, idx_last, prototypes, route_info)，
route_info['x_scores'] (1,N)、route_info['belong_idx'] (1,N)。对每个 TAB.dpr 注册 forward hook 抓取。

用法示例（x4，统一报告点 net_g_250000）：
    cd CATANet
    # Urban100：全量出 Fig.5 直方图 + 对 img_092/img_024 出 Fig.8 聚类图
    python scripts/build_routing_figures.py \
        --weights experiments/train_CATANet_x4_finetune/models/net_g_250000.pth \
        --scale 4 \
        --lr-dir /hy-tmp/TestDataSR/LR/LRBI/Urban100/x4 \
        --dataset-name Urban100 \
        --samples img_092,img_024 \
        --blocks 0,3,7 \
        --out-dir fig58_stage

    # Manga109：再跑一次（直方图按数据集分别出；Fig.8 加 ThatsIzumiko_000）
    python scripts/build_routing_figures.py \
        --weights experiments/train_CATANet_x4_finetune/models/net_g_250000.pth \
        --scale 4 \
        --lr-dir /hy-tmp/TestDataSR/LR/LRBI/Manga109/x4 \
        --dataset-name Manga109 \
        --samples ThatsIzumiko_000 \
        --blocks 0,3,7 \
        --out-dir fig58_stage

产物（--out-dir 下）回传到本地 paper/figures/ 后接入 LaTeX：
    fig5_<dataset>_xscore_hist.csv      每 block 直方图（bin_left,bin_right,count_b0..）
    fig5_<dataset>_xscore_stats.csv     每 block 统计 + router_scale + num_tokens + uniform_floor
    fig5_<dataset>_xscore_hist.png      预览（matplotlib 可用时）
    fig8_<dataset>_<sample>_b<idx>_cluster.png   聚类伪彩图（已 nearest 放大便于查看）
    fig8_<dataset>_<sample>_lr.png               对应 LR 输入（同口径）
"""
import argparse
import colorsys
import glob
import os
import os.path as osp

import cv2
import numpy as np
import torch

from basicsr.archs.catanet_arch import CATANet


def load_model(weights: str, scale: int, device: torch.device) -> CATANet:
    """构建与训练一致的默认 DPRNet 并加载权重（兼容 params / params_ema / 裸 state_dict）。"""
    model = CATANet(upscale=scale)
    ckpt = torch.load(weights, map_location='cpu')
    if isinstance(ckpt, dict) and 'params_ema' in ckpt:
        state = ckpt['params_ema']
    elif isinstance(ckpt, dict) and 'params' in ckpt:
        state = ckpt['params']
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def register_dpr_hooks(model: CATANet, captures: dict):
    """对每个 TAB.dpr 注册 forward hook，把当前前向的 x_scores / belong_idx 存入 captures。"""
    handles = []
    for i, block in enumerate(model.blocks):
        dpr = block[0].dpr

        def make_hook(idx):
            def hook(_module, _inp, out):
                route_info = out[3]
                captures[idx] = {
                    'x_scores': route_info['x_scores'][0].detach().float().cpu().numpy(),
                    'belong_idx': route_info['belong_idx'][0].detach().long().cpu().numpy(),
                }
            return hook

        handles.append(dpr.register_forward_hook(make_hook(i)))
    return handles


def read_lr_tensor(path: str, device: torch.device):
    """读 LR 图为模型输入张量 (1,3,H,W)，BGR->RGB、/255，与 BasicSR 测试口径一致。"""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'cannot read image: {path}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    return tensor, bgr


def find_sample_path(lr_dir: str, stem: str):
    """按 stem 匹配 LR 文件（兼容 img_092.png / img_092x4.png 等命名）。"""
    for cand in sorted(glob.glob(osp.join(lr_dir, f'{stem}*'))):
        if osp.isfile(cand):
            return cand
    return None


def cluster_palette(num_prototypes: int) -> np.ndarray:
    """生成 num_prototypes 个高区分度的离散 BGR 颜色（按黄金比错开 hue，相邻簇色差大）。"""
    lut = np.zeros((max(num_prototypes, 1), 3), dtype=np.uint8)
    for k in range(num_prototypes):
        hue = (k * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        lut[k] = [int(b * 255), int(g * 255), int(r * 255)]  # BGR for cv2.imwrite
    return lut


def colorize_clusters(belong_idx: np.ndarray, h: int, w: int, num_prototypes: int,
                      vis_scale: int) -> np.ndarray:
    lut = cluster_palette(num_prototypes)
    grid = belong_idx.reshape(h, w).clip(0, num_prototypes - 1)
    color = lut[grid]  # (h,w,3) BGR
    if vis_scale > 1:
        color = cv2.resize(color, (w * vis_scale, h * vis_scale),
                           interpolation=cv2.INTER_NEAREST)
    return color


@torch.no_grad()
def forward_capture(model, lr_tensor, captures):
    captures.clear()
    model(lr_tensor)


def accumulate_histogram(model, lr_dir, device, bins, hist_limit, captures):
    """遍历 lr_dir 累加每 block 的 x_scores 直方图与统计量。"""
    paths = sorted(p for p in glob.glob(osp.join(lr_dir, '*'))
                   if osp.isfile(p) and p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))
    if hist_limit > 0:
        paths = paths[:hist_limit]
    if not paths:
        raise FileNotFoundError(f'no images found under {lr_dir}')

    num_blocks = len(model.blocks)
    hist_counts = {i: np.zeros(len(bins) - 1, dtype=np.float64) for i in range(num_blocks)}
    sum_x = {i: 0.0 for i in range(num_blocks)}
    sumsq_x = {i: 0.0 for i in range(num_blocks)}
    n_tokens = {i: 0 for i in range(num_blocks)}
    all_vals = {i: [] for i in range(num_blocks)}  # for median / percentiles

    for k, path in enumerate(paths):
        lr_tensor, _ = read_lr_tensor(path, device)
        forward_capture(model, lr_tensor, captures)
        for i in range(num_blocks):
            xs = captures[i]['x_scores']
            hist_counts[i] += np.histogram(xs, bins=bins)[0]
            sum_x[i] += float(xs.sum())
            sumsq_x[i] += float((xs.astype(np.float64) ** 2).sum())
            n_tokens[i] += xs.size
            all_vals[i].append(xs)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'  [hist] {k + 1}/{len(paths)}  {osp.basename(path)}')

    return hist_counts, sum_x, sumsq_x, n_tokens, all_vals, len(paths)


def block_router_scales(model):
    scales = {}
    for i, block in enumerate(model.blocks):
        dpr = block[0].dpr
        s = dpr.router_logit_scale.detach().exp().clamp(max=dpr.max_router_logit_scale)
        scales[i] = float(s.cpu())
    return scales


def write_histogram_csv(out_dir, dataset, bins, hist_counts):
    num_blocks = len(hist_counts)
    path = osp.join(out_dir, f'fig5_{dataset}_xscore_hist.csv')
    header = ['bin_left', 'bin_right'] + [f'count_b{i}' for i in range(num_blocks)]
    with open(path, 'w') as f:
        f.write(','.join(header) + '\n')
        for j in range(len(bins) - 1):
            row = [f'{bins[j]:.4f}', f'{bins[j + 1]:.4f}']
            row += [f'{int(hist_counts[i][j])}' for i in range(num_blocks)]
            f.write(','.join(row) + '\n')
    return path


def write_stats_csv(out_dir, dataset, model, sum_x, sumsq_x, n_tokens, all_vals, scales):
    path = osp.join(out_dir, f'fig5_{dataset}_xscore_stats.csv')
    header = ['block', 'num_tokens', 'uniform_floor', 'mean', 'std', 'median',
              'p10', 'p90', 'frac_above_uniform', 'router_scale']
    with open(path, 'w') as f:
        f.write(','.join(header) + '\n')
        for i in range(len(model.blocks)):
            m = model.num_tokens[i]
            floor = 1.0 / m
            mean = sum_x[i] / max(n_tokens[i], 1)
            var = sumsq_x[i] / max(n_tokens[i], 1) - mean ** 2
            std = float(np.sqrt(max(var, 0.0)))
            vals = np.concatenate(all_vals[i]) if all_vals[i] else np.zeros(1)
            median = float(np.median(vals))
            p10 = float(np.percentile(vals, 10))
            p90 = float(np.percentile(vals, 90))
            frac_above = float((vals > floor).mean())
            f.write(','.join([
                str(i), str(m), f'{floor:.6f}', f'{mean:.6f}', f'{std:.6f}',
                f'{median:.6f}', f'{p10:.6f}', f'{p90:.6f}',
                f'{frac_above:.6f}', f'{scales[i]:.4f}',
            ]) + '\n')
    return path


def try_plot_histogram(out_dir, dataset, bins, hist_counts, model, scales, blocks):
    """matplotlib 可用时出预览 PNG；不可用则跳过（仅打印提示，不报错）。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f'[warn] matplotlib unavailable, skip PNG preview ({e}); use CSV + pgfplots locally.')
        return None

    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in blocks:
        counts = hist_counts[i]
        total = counts.sum()
        dens = counts / total if total > 0 else counts
        ax.plot(centers, dens, label=f'block {i} (M={model.num_tokens[i]}, $\\tau$={scales[i]:.2f})')
        floor = 1.0 / model.num_tokens[i]
        ax.axvline(floor, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax.set_xlabel('token confidence $x^{s}$ (= max softmax score)')
    ax.set_ylabel('normalized frequency')
    ax.set_title(f'Routing confidence distribution — {dataset}')
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = osp.join(out_dir, f'fig5_{dataset}_xscore_hist.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def build_cluster_figures(model, lr_dir, dataset, samples, blocks, device,
                          out_dir, vis_scale, captures):
    for stem in samples:
        path = find_sample_path(lr_dir, stem)
        if path is None:
            print(f'[warn] sample not found, skip: {stem} (under {lr_dir})')
            continue
        lr_tensor, lr_bgr = read_lr_tensor(path, device)
        _, _, h, w = lr_tensor.shape
        forward_capture(model, lr_tensor, captures)

        lr_vis = cv2.resize(lr_bgr, (w * vis_scale, h * vis_scale),
                            interpolation=cv2.INTER_NEAREST)
        lr_out = osp.join(out_dir, f'fig8_{dataset}_{stem}_lr.png')
        cv2.imwrite(lr_out, lr_vis)
        print(f'  [cluster] {dataset}/{stem}: LR {w}x{h} -> {lr_out}')

        for i in blocks:
            belong = captures[i]['belong_idx']
            assert belong.size == h * w, (
                f'block{i}: belong_idx size {belong.size} != H*W {h * w}')
            color = colorize_clusters(belong, h, w, model.num_tokens[i], vis_scale)
            cl_out = osp.join(out_dir, f'fig8_{dataset}_{stem}_b{i}_cluster.png')
            cv2.imwrite(cl_out, color)
            n_active = len(np.unique(belong))
            print(f'      block {i}: M={model.num_tokens[i]}, active={n_active} -> {cl_out}')
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description='Generate Fig.5 (x_scores histogram) and Fig.8 (routing cluster maps).')
    parser.add_argument('--weights', required=True, help='checkpoint path (.pth)')
    parser.add_argument('--scale', type=int, default=4, choices=[2, 3, 4])
    parser.add_argument('--lr-dir', required=True, help='LR image dir for one dataset (e.g. .../LRBI/Urban100/x4)')
    parser.add_argument('--dataset-name', default=None, help='dataset label for filenames (default: infer from lr-dir)')
    parser.add_argument('--samples', default='', help='comma-separated stems for Fig.8 cluster maps (empty=skip Fig.8)')
    parser.add_argument('--blocks', default='0,3,7', help='comma-separated TAB block indices to visualize')
    parser.add_argument('--out-dir', default='fig58_stage', help='output dir for assets to ship back')
    parser.add_argument('--bins', type=int, default=50, help='histogram bins over [0,1] for Fig.5')
    parser.add_argument('--hist-limit', type=int, default=0, help='cap #images for Fig.5 (0=all in lr-dir)')
    parser.add_argument('--vis-scale', type=int, default=4, help='nearest upscale for cluster/LR preview pngs')
    parser.add_argument('--skip-hist', action='store_true', help='only build Fig.8 cluster maps, skip Fig.5')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = args.dataset_name or osp.basename(osp.dirname(args.lr_dir.rstrip('/')))
    blocks = [int(b) for b in args.blocks.split(',') if b.strip() != '']
    samples = [s.strip() for s in args.samples.split(',') if s.strip() != '']
    os.makedirs(args.out_dir, exist_ok=True)

    print('=' * 64)
    print(f'Model    : DPRNet (CATANet) x{args.scale}')
    print(f'Weights  : {args.weights}')
    print(f'Dataset  : {dataset}  ({args.lr_dir})')
    print(f'Blocks   : {blocks}   Samples: {samples or "(none)"}')
    print(f'Out dir  : {args.out_dir}   Device: {device}')
    print('=' * 64)

    model = load_model(args.weights, args.scale, device)
    for b in blocks:
        if b < 0 or b >= len(model.blocks):
            parser.error(f'--blocks index {b} out of range [0,{len(model.blocks) - 1}]')
    captures = {}
    handles = register_dpr_hooks(model, captures)
    scales = block_router_scales(model)

    try:
        if not args.skip_hist:
            bins = np.linspace(0.0, 1.0, args.bins + 1)
            print('[Fig.5] accumulating x_scores histogram ...')
            hist_counts, sum_x, sumsq_x, n_tokens, all_vals, n_imgs = accumulate_histogram(
                model, args.lr_dir, device, bins, args.hist_limit, captures)
            csv1 = write_histogram_csv(args.out_dir, dataset, bins, hist_counts)
            csv2 = write_stats_csv(args.out_dir, dataset, model, sum_x, sumsq_x,
                                   n_tokens, all_vals, scales)
            png = try_plot_histogram(args.out_dir, dataset, bins, hist_counts, model, scales, blocks)
            print(f'[Fig.5] {n_imgs} imgs -> {csv1}')
            print(f'[Fig.5] stats        -> {csv2}')
            if png:
                print(f'[Fig.5] preview      -> {png}')

        if samples:
            print('[Fig.8] building cluster maps ...')
            build_cluster_figures(model, args.lr_dir, dataset, samples, blocks,
                                  device, args.out_dir, args.vis_scale, captures)
        else:
            print('[Fig.8] no --samples given, skipped.')
    finally:
        for h in handles:
            h.remove()

    print('=' * 64)
    print(f'Done. Assets in: {args.out_dir}')
    print('Ship back to local paper/figures/ and wire into LaTeX.')
    print('=' * 64)


if __name__ == '__main__':
    main()
