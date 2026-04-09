#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合策略：结合相似性分数和差异矩阵进行标签划分（修复索引版本）
- 计算相似性指标（SSIM + 相关系数）
- 计算差异指标（MAE, RMSE, 最大差异等）
- 综合两类指标进行更精准的分类
- 标签含义：0=相似且差异小, 1=不相似且差异大
- 修复：确保DataFrame索引始终连续
"""
import os
import json
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
try:
    import cooler
except Exception:
    cooler = None
# =====================
# 混合策略配置参数
# =====================
CONFIG = {
    "mcool1": "K562_4DNFI18UHVRO.mcool::/resolutions/5000",
    "mcool2": "GM12878_4DNFIXP4QG5B.mcool::/resolutions/5000",
    "out_npz": "K562-GM12878.npz",
    "save_coords": True,
    # 基本参数
    "bin_bp": 5000,
    "window_bp": 500000,
    "step_bp": 250000,
    "chroms": ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7",
               "chr8", "chr9", "chr10", "chr11", "chr12", "chr13",
               "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
               "chr20", "chr21", "chr22"],
    # 混合策略参数
    "hybrid_method": "weighted_combination", # "weighted_combination", "dual_threshold", "decision_tree"
   
    # 相似性指标权重和参数
    "similarity_alpha": 0.6, # SSIM权重
    "similarity_beta": 0.4, # 皮尔逊相关权重
   
    # 差异指标权重
    "diff_mae_weight": 0.4,
    "diff_rmse_weight": 0.3,
    "diff_max_weight": 0.2,
    "diff_std_weight": 0.1,
   
    # 综合评估权重
    "similarity_component_weight": 0.5, # 相似性组件权重
    "difference_component_weight": 0.5, # 差异组件权重
   
    # 阈值策略
    "threshold_method": "adaptive_quantiles", # "adaptive_quantiles", "fixed_dual", "consensus"
   
    # 自适应分位数阈值（修改：调整以增加分类平衡）
    "high_dissim_percentile": 90, # 高不相似阈值（从80降低到75）
    "low_dissim_percentile": 10, # 低不相似阈值（从20增加到25）
    "high_diff_percentile": 90, # 高差异阈值（从80降低）
    "low_diff_percentile": 10, # 低差异阈值（从20增加）
   
    # 固定双阈值
    "fixed_similarity_high": 0.95, # 高相似阈值
    "fixed_similarity_low": 0.90, # 低相似阈值
    "fixed_difference_high": 0.002, # 高差异阈值
    "fixed_difference_low": 0.0008, # 低差异阈值
   
    # 质量控制
    "outlier_removal": True,
    "outlier_z_threshold": 3.0,
    "min_samples_per_class": 100,
   
    # 矩阵处理
    "matrix_balance": True,
    "matrix_log1p": True,
    "matrix_clip_percentile": 99.5
}
# =====================
# 混合评估工具函数（保持不变）
# =====================
def calculate_similarity_metrics(mat1: np.ndarray, mat2: np.ndarray,
                               alpha: float = 0.6, beta: float = 0.4) -> Dict[str, float]:
    """
    计算相似性指标（保持原有逻辑）
    """
    # 1. SSIM结构相似性
    ssim_score, ssim_map = ssim(mat1, mat2, full=True)
   
    # 2. 皮尔逊相关系数（上三角区域）
    triu_indices = np.triu_indices_from(mat1, k=1)
    vec1, vec2 = mat1[triu_indices], mat2[triu_indices]
   
    if np.std(vec1) == 0 or np.std(vec2) == 0:
        pearson_score = 0.0
    else:
        pearson_score, _ = pearsonr(vec1, vec2)
        if np.isnan(pearson_score):
            pearson_score = 0.0
   
    # 3. 综合相似性分数
    combined_similarity = alpha * ssim_score + beta * pearson_score
   
    # 4. 额外相似性指标
    # 余弦相似度
    flat1, flat2 = mat1.flatten(), mat2.flatten()
    cosine_sim = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2) + 1e-8)
   
    # 归一化相互信息（简化版）
    hist_2d, _, _ = np.histogram2d(flat1, flat2, bins=20)
    hist_2d = hist_2d / np.sum(hist_2d)
    pxy = hist_2d + 1e-10 # 避免log(0)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
   
    # 互信息计算
    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
   
    # 归一化
    hx = -np.sum(px * np.log2(px))
    hy = -np.sum(py * np.log2(py))
    nmi = 2 * mi / (hx + hy) if (hx + hy) > 0 else 0
   
    return {
        'ssim': ssim_score,
        'pearson': pearson_score,
        'combined_similarity': combined_similarity,
        'cosine_similarity': cosine_sim,
        'normalized_mutual_info': nmi,
        'ssim_map_variance': np.var(ssim_map)
    }
def calculate_difference_metrics(mat1: np.ndarray, mat2: np.ndarray,
                               weights: Dict[str, float] = None) -> Dict[str, float]:
    """
    计算差异指标
    """
    if weights is None:
        weights = {'mae': 0.4, 'rmse': 0.3, 'max': 0.2, 'std': 0.1}
   
    # 差异矩阵
    diff_matrix = mat1 - mat2
    abs_diff = np.abs(diff_matrix)
   
    # 基础差异指标
    mae = np.mean(abs_diff)
    rmse = np.sqrt(np.mean(diff_matrix ** 2))
    max_diff = np.max(abs_diff)
    std_diff = np.std(diff_matrix)
   
    # 综合差异分数
    combined_difference = (weights['mae'] * mae +
                          weights['rmse'] * rmse +
                          weights['max'] * max_diff +
                          weights['std'] * std_diff)
   
    # 上三角区域差异
    triu_indices = np.triu_indices_from(diff_matrix, k=1)
    triu_mae = np.mean(abs_diff[triu_indices])
    triu_rmse = np.sqrt(np.mean(diff_matrix[triu_indices] ** 2))
   
    # 分位数差异
    p75_diff = np.percentile(abs_diff, 75)
    p90_diff = np.percentile(abs_diff, 90)
    p95_diff = np.percentile(abs_diff, 95)
   
    # 相对差异
    epsilon = 1e-8
    relative_mae = np.mean(abs_diff / (np.maximum(mat1, mat2) + epsilon))
   
    # 空间差异模式分析
    # 中心vs边缘差异
    center = mat1.shape[0] // 4
    center_slice = slice(center, -center)
    center_diff = np.mean(abs_diff[center_slice, center_slice])
    edge_diff = np.mean(abs_diff) - center_diff
   
    # 对角线vs非对角线差异
    diag_diff = np.mean(np.abs(np.diag(diff_matrix)))
    offdiag_diff = (np.sum(abs_diff) - np.sum(np.abs(np.diag(diff_matrix)))) / (abs_diff.size - len(np.diag(diff_matrix)))
   
    return {
        'mae': mae,
        'rmse': rmse,
        'max_diff': max_diff,
        'std_diff': std_diff,
        'combined_difference': combined_difference,
        'triu_mae': triu_mae,
        'triu_rmse': triu_rmse,
        'p75_diff': p75_diff,
        'p90_diff': p90_diff,
        'p95_diff': p95_diff,
        'relative_mae': relative_mae,
        'center_diff': center_diff,
        'edge_diff': edge_diff,
        'center_edge_ratio': center_diff / (edge_diff + epsilon),
        'diag_diff': diag_diff,
        'offdiag_diff': offdiag_diff,
        'diag_offdiag_ratio': diag_diff / (offdiag_diff + epsilon)
    }
def compute_hybrid_score(similarity_metrics: Dict[str, float],
                        difference_metrics: Dict[str, float],
                        method: str = "weighted_combination",
                        sim_weight: float = 0.5,
                        diff_weight: float = 0.5) -> Dict[str, float]:
    """
    计算混合评估分数
    """
    # 相似性分量（取反转为不相似性）
    similarity_component = 1 - similarity_metrics['combined_similarity']
   
    # 差异分量
    difference_component = difference_metrics['combined_difference']
   
    if method == "weighted_combination":
        # 加权组合
        hybrid_score = sim_weight * similarity_component + diff_weight * difference_component
       
        return {
            'hybrid_score': hybrid_score,
            'similarity_component': similarity_component,
            'difference_component': difference_component,
            'method': method
        }
       
    elif method == "geometric_mean":
        # 几何平均（避免某个分量过小）
        hybrid_score = np.sqrt(similarity_component * difference_component)
       
        return {
            'hybrid_score': hybrid_score,
            'similarity_component': similarity_component,
            'difference_component': difference_component,
            'method': method
        }
       
    elif method == "max_consensus":
        # 最大一致性：两个分量都高时才认为是高差异
        hybrid_score = min(similarity_component, difference_component)
       
        return {
            'hybrid_score': hybrid_score,
            'similarity_component': similarity_component,
            'difference_component': difference_component,
            'method': method
        }
   
    else:
        return compute_hybrid_score(similarity_metrics, difference_metrics,
                                  "weighted_combination", sim_weight, diff_weight)
def assign_hybrid_labels(df: pd.DataFrame, method: str = "adaptive_quantiles", **kwargs) -> np.ndarray:
    """
    基于混合指标分配标签
    """
    n_samples = len(df)
    labels = np.full(n_samples, -1) # 初始化为未分类
   
    if method == "adaptive_quantiles":
        # 自适应分位数阈值
        hybrid_scores = df['hybrid_score'].values
        sim_components = df['similarity_component'].values
        diff_components = df['difference_component'].values
       
        # 混合分数阈值
        hybrid_high = np.percentile(hybrid_scores, kwargs.get('high_hybrid_percentile', 80))
        hybrid_low = np.percentile(hybrid_scores, kwargs.get('low_hybrid_percentile', 20))
       
        # 分配标签
        labels[hybrid_scores >= hybrid_high] = 1 # 高不相似高差异
        labels[hybrid_scores <= hybrid_low] = 0 # 高相似低差异
       
        print(f"自适应阈值: 相似低差异≤{hybrid_low:.6f}, 不相似高差异≥{hybrid_high:.6f}")
       
    elif method == "dual_threshold":
        # 双重阈值：同时满足相似性和差异性条件
        sim_components = df['similarity_component'].values
        diff_components = df['difference_component'].values
       
        sim_high = np.percentile(sim_components, kwargs.get('high_dissim_percentile', 80))
        sim_low = np.percentile(sim_components, kwargs.get('low_dissim_percentile', 20))
        diff_high = np.percentile(diff_components, kwargs.get('high_diff_percentile', 80))
        diff_low = np.percentile(diff_components, kwargs.get('low_diff_percentile', 20))
       
        # 严格标准：两个指标都要满足
        high_mask = (sim_components >= sim_high) & (diff_components >= diff_high)
        low_mask = (sim_components <= sim_low) & (diff_components <= diff_low)
       
        labels[high_mask] = 1
        labels[low_mask] = 0
       
        print(f"双重阈值:")
        print(f" 相似性: 低≤{sim_low:.6f}, 高≥{sim_high:.6f}")
        print(f" 差异性: 低≤{diff_low:.6f}, 高≥{diff_high:.6f}")
       
    elif method == "consensus":
        # 共识方法：多个指标的一致性投票
        hybrid_scores = df['hybrid_score'].values
        sim_components = df['similarity_component'].values
        diff_components = df['difference_component'].values
       
        # 每个指标独立投票
        hybrid_vote = np.zeros(n_samples)
        hybrid_vote[hybrid_scores >= np.percentile(hybrid_scores, 75)] = 1
        hybrid_vote[hybrid_scores <= np.percentile(hybrid_scores, 25)] = -1
       
        sim_vote = np.zeros(n_samples)
        sim_vote[sim_components >= np.percentile(sim_components, 75)] = 1
        sim_vote[sim_components <= np.percentile(sim_components, 25)] = -1
       
        diff_vote = np.zeros(n_samples)
        diff_vote[diff_components >= np.percentile(diff_components, 75)] = 1
        diff_vote[diff_components <= np.percentile(diff_components, 25)] = -1
       
        # 综合投票
        total_votes = hybrid_vote + sim_vote + diff_vote
       
        labels[total_votes >= 2] = 1 # 至少2票赞成高差异
        labels[total_votes <= -2] = 0 # 至少2票赞成低差异
       
        print(f"共识投票: 高差异={np.sum(labels==1)}, 低差异={np.sum(labels==0)}")
       
    elif method == "fixed_dual":
        # 固定双阈值
        sim_scores = 1 - df['similarity_component'].values # 转回相似性
        diff_scores = df['difference_component'].values
       
        sim_high = kwargs.get('fixed_similarity_high', 0.95)
        sim_low = kwargs.get('fixed_similarity_low', 0.90)
        diff_high = kwargs.get('fixed_difference_high', 0.002)
        diff_low = kwargs.get('fixed_difference_low', 0.0008)
       
        # 高相似低差异
        high_sim_low_diff = (sim_scores >= sim_high) & (diff_scores <= diff_low)
        # 低相似高差异
        low_sim_high_diff = (sim_scores <= sim_low) & (diff_scores >= diff_high)
       
        labels[high_sim_low_diff] = 0
        labels[low_sim_high_diff] = 1
       
        print(f"固定双阈值:")
        print(f" 相似性: {sim_low} - {sim_high}")
        print(f" 差异性: {diff_low} - {diff_high}")
   
    # 统计结果
    n_high = np.sum(labels == 1)
    n_low = np.sum(labels == 0)
    n_unclassified = np.sum(labels == -1)
   
    print(f"分类结果: 不相似高差异={n_high}, 相似低差异={n_low}, 未分类={n_unclassified}")
    print(f"分类率: {(n_high + n_low) / len(labels) * 100:.1f}%")
   
    # 新增：确保每个类至少有最小样本数（通过调整阈值）
    min_samples = kwargs.get('min_samples_per_class', 10)  # 默认10，可从CONFIG传
    if n_low < min_samples or n_high < min_samples:
        print(f"警告: 标签0样本={n_low}, 标签1样本={n_high}，不足{min_samples}。调整阈值重分配...")
        # 示例调整：降低高阈值，增加低阈值
        if method == "adaptive_quantiles":
            hybrid_high = np.percentile(hybrid_scores, 70)  # 降低到70
            hybrid_low = np.percentile(hybrid_scores, 30)   # 增加到30
            labels = np.full(n_samples, -1)
            labels[hybrid_scores >= hybrid_high] = 1
            labels[hybrid_scores <= hybrid_low] = 0
            n_high = np.sum(labels == 1)
            n_low = np.sum(labels == 0)
            print(f"调整后: 不相似高差异={n_high}, 相似低差异={n_low}")
        # 可以根据其他method添加类似调整逻辑
   
    return labels
# =====================
# 原有工具函数（保持不变）
# =====================
def _expand(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    return os.path.abspath(os.path.expanduser(os.path.expandvars(p)))
def _normalize_mcool_uri(uri_or_path: str, bin_bp: int) -> str:
    if uri_or_path is None:
        raise ValueError("mcool 路径不能为空")
    if "::" in uri_or_path:
        base, group = uri_or_path.split("::", 1)
        base = _expand(base)
        if not os.path.exists(base):
            raise FileNotFoundError(f"mcool 文件不存在: {base}")
        return f"{base}::{group}"
    else:
        base = _expand(uri_or_path)
        if not os.path.exists(base):
            raise FileNotFoundError(f"mcool 文件不存在: {base}")
        group = f"/resolutions/{int(bin_bp)}"
        return f"{base}::{group}"
def load_chrom_sizes_from_cool(path: str, whitelist: Optional[List[str]] = None) -> List[Tuple[str, int]]:
    if cooler is None:
        raise RuntimeError("需要安装 cooler：pip install cooler")
    c = cooler.Cooler(path)
    chroms = c.chromnames
    sizes = c.chromsizes
    out = []
    for ch in chroms:
        if whitelist and ch not in whitelist:
            continue
        out.append((ch, int(sizes[ch])))
    return out
def fetch_dense_matrix(cool_obj, region: str, balance: bool = True) -> Optional[np.ndarray]:
    mat = cool_obj.matrix(balance=balance, sparse=False).fetch(region)
    mat = np.asarray(mat, dtype=np.float32)
    # 检查 NaN 或零值比例
    nan_ratio = np.isnan(mat).mean()
    zero_ratio = (mat == 0).mean()
    gap_threshold = 0.1 # 自定义阈值，例如 10% 的 NaN 或零值视为 gap
    if nan_ratio > gap_threshold or zero_ratio > gap_threshold:
        return None
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat
   
def postprocess_matrix(mat: np.ndarray, log1p: bool, clip_pct: Optional[float]) -> np.ndarray:
    x = mat.copy()
    if clip_pct is not None:
        hi = np.percentile(x, clip_pct)
        if hi > 0:
            x = np.clip(x, a_min=0, a_max=hi)
    if log1p:
        x = np.log1p(x)
    return x.astype(np.float32)
# =====================
# 修复后的混合策略主函数
# =====================
def run_hybrid_strategy(cfg: dict):
    """
    混合策略主函数（修复索引版本）
    """
    bin_bp = int(cfg["bin_bp"])
    mcool1 = _normalize_mcool_uri(cfg.get("mcool1"), bin_bp)
    mcool2 = _normalize_mcool_uri(cfg.get("mcool2"), bin_bp)
    out_npz = _expand(cfg.get("out_npz"))
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    print("[混合策略] 相似性+差异矩阵综合标签分配（修复索引版本）")
    print(" mcool1:", mcool1)
    print(" mcool2:", mcool2)
    print(" out_npz:", out_npz)
    # 打开 cool 文件
    c1 = cooler.Cooler(mcool1)
    c2 = cooler.Cooler(mcool2)
    chrom_sizes = load_chrom_sizes_from_cool(mcool1, whitelist=cfg.get("chroms", None))
    # 参数提取
    window_bp = int(cfg["window_bp"])
    step_bp = int(cfg["step_bp"])
    matrix_balance = bool(cfg.get("matrix_balance", True))
    matrix_log1p = bool(cfg.get("matrix_log1p", True))
    matrix_clip_pct = cfg.get("matrix_clip_percentile", 99.5)
    save_coords = bool(cfg.get("save_coords", True))
   
    # 混合策略参数
    hybrid_method = cfg.get("hybrid_method", "weighted_combination")
    threshold_method = cfg.get("threshold_method", "adaptive_quantiles")
    similarity_alpha = cfg.get("similarity_alpha", 0.6)
    similarity_beta = cfg.get("similarity_beta", 0.4)
   
    diff_weights = {
        'mae': cfg.get('diff_mae_weight', 0.4),
        'rmse': cfg.get('diff_rmse_weight', 0.3),
        'max': cfg.get('diff_max_weight', 0.2),
        'std': cfg.get('diff_std_weight', 0.1)
    }
   
    sim_weight = cfg.get("similarity_component_weight", 0.5)
    diff_weight = cfg.get("difference_component_weight", 0.5)
    # 第一遍扫描：计算混合指标
    print("\n第一遍扫描：计算相似性和差异指标...")
    rows = []
   
    for chrom, clen in chrom_sizes:
        print(f"处理染色体 {chrom}...")
        for start in range(0, clen - window_bp, step_bp):
            end = start + window_bp
            region = f"{chrom}:{start}-{end}"
           
            try:
                m1 = fetch_dense_matrix(c1, region, balance=matrix_balance)
                m2 = fetch_dense_matrix(c2, region, balance=matrix_balance)
                if m1 is None or m2 is None:
                    continue
            except Exception:
                continue
           
            if m1.shape != m2.shape or m1.shape[0] == 0:
                continue
            # 矩阵后处理
            m1 = postprocess_matrix(m1, log1p=matrix_log1p, clip_pct=matrix_clip_pct)
            m2 = postprocess_matrix(m2, log1p=matrix_log1p, clip_pct=matrix_clip_pct)
            # 计算相似性指标
            similarity_metrics = calculate_similarity_metrics(m1, m2, similarity_alpha, similarity_beta)
           
            # 计算差异指标
            difference_metrics = calculate_difference_metrics(m1, m2, diff_weights)
           
            # 计算混合分数
            hybrid_metrics = compute_hybrid_score(similarity_metrics, difference_metrics,
                                                hybrid_method, sim_weight, diff_weight)
            # 组装数据行
            row_data = {
                "chrom": chrom,
                "start": start,
                "end": end,
                "region": region
            }
           
            # 添加所有指标
            row_data.update(similarity_metrics)
            row_data.update(difference_metrics)
            row_data.update(hybrid_metrics)
           
            rows.append(row_data)
    if not rows:
        raise RuntimeError("未生成任何有效窗口")
    df = pd.DataFrame(rows)
    print(f"\n混合指标计算完成，共{len(df)}个样本")
    print(f"初始DataFrame索引范围: {df.index.min()} - {df.index.max()}")
   
    # 打印统计摘要
    print(f"\n指标统计摘要:")
    print(f"相似性分数: {df['combined_similarity'].mean():.6f} ± {df['combined_similarity'].std():.6f}")
    print(f"差异分数: {df['combined_difference'].mean():.6f} ± {df['combined_difference'].std():.6f}")
    print(f"混合分数: {df['hybrid_score'].mean():.6f} ± {df['hybrid_score'].std():.6f}")
    # 异常值处理 - 关键修复：重置索引
    if cfg.get('outlier_removal', False):
        z_scores = np.abs(stats.zscore(df['hybrid_score']))
        outlier_mask = z_scores > cfg.get('outlier_z_threshold', 3.0)
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            print(f"移除 {n_outliers} 个异常值")
            df = df[~outlier_mask].reset_index(drop=True) # 关键修复：重置索引
            print(f"异常值移除后DataFrame索引范围: {df.index.min()} - {df.index.max()}")
    # 标签分配
    print(f"\n使用 {threshold_method} 方法分配混合标签...")
   
    threshold_kwargs = {
        'high_hybrid_percentile': cfg.get('high_dissim_percentile', 75),  # 修改匹配CONFIG
        'low_hybrid_percentile': cfg.get('low_dissim_percentile', 25),
        'high_dissim_percentile': cfg.get('high_dissim_percentile', 75),
        'low_dissim_percentile': cfg.get('low_dissim_percentile', 25),
        'high_diff_percentile': cfg.get('high_diff_percentile', 75),
        'low_diff_percentile': cfg.get('low_diff_percentile', 25),
        'fixed_similarity_high': cfg.get('fixed_similarity_high', 0.95),
        'fixed_similarity_low': cfg.get('fixed_similarity_low', 0.90),
        'fixed_difference_high': cfg.get('fixed_difference_high', 0.002),
        'fixed_difference_low': cfg.get('fixed_difference_low', 0.0008),
        'min_samples_per_class': cfg.get('min_samples_per_class', 100)  # 新增传入
    }
    labels = assign_hybrid_labels(df, method=threshold_method, **threshold_kwargs)
    df['label'] = labels
    # 筛选分类成功的样本 - 关键修复：重置索引
    classified_df = df[df['label'].isin([0, 1])].copy().reset_index(drop=True) # 关键修复：重置索引
    print(f"筛选后DataFrame索引范围: {classified_df.index.min()} - {classified_df.index.max()}")
   
    if len(classified_df) == 0:
        raise RuntimeError("没有样本被成功分类")
    # 检查每个类别的最小样本数
    min_samples = cfg.get('min_samples_per_class', 100)
    label_counts = classified_df['label'].value_counts()
   
    for label_val, count in label_counts.items():
        if count < min_samples:
            print(f"警告: 标签{label_val}只有{count}个样本，少于最小要求{min_samples}")
    print(f"\n最终分类结果:")
    print(f"标签0 (相似低差异): {(classified_df['label'] == 0).sum()} 个样本")
    print(f"标签1 (不相似高差异): {(classified_df['label'] == 1).sum()} 个样本")
    # 第二遍扫描：重新提取分类样本的矩阵数据 - 关键修复：使用连续索引遍历
    print(f"\n第二遍扫描：提取分类样本的矩阵数据...")
    mats1, mats2, final_labels, coords = [], [], [], []
    # 关键修复：使用iloc按位置遍历，而不是iterrows
    for i in range(len(classified_df)):
        row = classified_df.iloc[i] # 使用iloc确保按位置访问
        region = row['region']
       
        try:
            m1 = fetch_dense_matrix(c1, region, balance=matrix_balance)
            m2 = fetch_dense_matrix(c2, region, balance=matrix_balance)
           
            if m1 is None or m2 is None:
                print(f"样本 {i}: 区域 {region} 矩阵获取失败，跳过")
                continue
               
            if m1.shape != m2.shape:
                minB = min(m1.shape[0], m2.shape[0])
                m1 = m1[:minB, :minB]
                m2 = m2[:minB, :minB]
           
            m1 = postprocess_matrix(m1, log1p=matrix_log1p, clip_pct=matrix_clip_pct)
            m2 = postprocess_matrix(m2, log1p=matrix_log1p, clip_pct=matrix_clip_pct)
            mats1.append(m1)
            mats2.append(m2)
            final_labels.append(int(row["label"]))
            if save_coords:
                coords.append(region)
               
        except Exception as e:
            print(f"样本 {i} 处理出错: {e}，跳过")
            continue
    # 验证数据一致性
    if len(mats1) != len(mats2) or len(mats1) != len(final_labels):
        print(f"警告：数组长度不一致: mats1={len(mats1)}, mats2={len(mats2)}, labels={len(final_labels)}")
        min_len = min(len(mats1), len(mats2), len(final_labels))
        print(f"截取至最小长度: {min_len}")
        mats1 = mats1[:min_len]
        mats2 = mats2[:min_len]
        final_labels = final_labels[:min_len]
        if save_coords and len(coords) > min_len:
            coords = coords[:min_len]
    if len(mats1) == 0:
        raise RuntimeError("第二遍扫描未生成任何有效的矩阵数据")
    # 保存数据
    X1 = np.stack(mats1, axis=0)
    X2 = np.stack(mats2, axis=0)
    y = np.asarray(final_labels, dtype=np.int8)
    print(f"最终矩阵形状: X1={X1.shape}, X2={X2.shape}, y={y.shape}")
    save_dict = {"X1": X1, "X2": X2, "y": y}
    if save_coords and coords:
        save_dict["coords"] = np.asarray(coords)
   
    np.savez_compressed(out_npz, **save_dict)
   
    summ = {
        "saved_npz": out_npz,
        "X_shape": X1.shape,
        "n_label0_similar": int((y == 0).sum()),
        "n_label1_different": int((y == 1).sum())
    }
    print(json.dumps(summ, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    run_hybrid_strategy(CONFIG)