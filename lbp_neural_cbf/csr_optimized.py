"""
优化后的CSR模块 - 针对大规模子区域验证

这个模块实现了针对论文中大规模子区域验证的CSR方案优化：
1. 增量协方差计算
2. 统计采样策略
3. 与LBP验证流程的高效集成
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import random


@dataclass
class RegionStats:
    """区域统计信息"""
    total_verified: int = 0
    total_failed: int = 0
    verified_A_L: np.ndarray = None
    failed_A_L: np.ndarray = None


class IncrementalCovariance:
    """
    增量协方差计算类，专门设计用于处理数百万个区域。
    """

    def __init__(self, d: int):
        self.d = d
        self.n_V = 0  # 已验证区域计数
        self.n_F = 0  # 失败区域计数
        self.M_V = np.zeros((d, d))  # 已验证区域协方差矩阵
        self.M_F = np.zeros((d, d))  # 失败区域协方差矩阵

    def update_verified(self, A_L: np.ndarray):
        """
        增量更新已验证区域的协方差矩阵。

        Args:
            A_L: [d]或[1,d]形状的线性边界系数
        """
        A_L = A_L.flatten()
        A_L_reshaped = A_L.reshape(1, -1)  # 确保2D
        self.n_V += 1
        if self.n_V == 1:
            self.M_V = A_L_reshaped.T @ A_L_reshaped
        else:
            old_mean = self.M_V
            delta = A_L_reshaped.T @ A_L_reshaped - old_mean
            self.M_V = old_mean + delta / self.n_V

    def update_failed(self, A_L: np.ndarray):
        """
        增量更新失败区域的协方差矩阵。

        Args:
            A_L: [d]或[1,d]形状的线性边界系数
        """
        A_L = A_L.flatten()
        A_L_reshaped = A_L.reshape(1, -1)  # 确保2D
        self.n_F += 1
        if self.n_F == 1:
            self.M_F = A_L_reshaped.T @ A_L_reshaped
        else:
            old_mean = self.M_F
            delta = A_L_reshaped.T @ A_L_reshaped - old_mean
            self.M_F = old_mean + delta / self.n_F

    def get_covariance(self):
        """获取当前协方差矩阵"""
        if self.n_V == 0 or self.n_F == 0:
            raise ValueError("需要至少一个已验证和一个失败的区域")
        return self.M_V / self.n_V, self.M_F / self.n_F

    def reset(self):
        """重置计数器和协方差矩阵"""
        self.n_V = 0
        self.n_F = 0
        self.M_V = np.zeros((self.d, self.d))
        self.M_F = np.zeros((self.d, self.d))


class StatisticalSampler:
    """
    统计采样器，用于处理超大规模区域。
    """

    def __init__(self, d: int, sample_ratio: float = 0.01):
        self.d = d
        self.sample_ratio = sample_ratio
        self.verified_samples: List[np.ndarray] = []
        self.failed_samples: List[np.ndarray] = []

    def add_region(self, A_L: np.ndarray, is_verified: bool):
        """添加一个区域到采样器"""
        if random.random() < self.sample_ratio:
            if is_verified:
                self.verified_samples.append(A_L.flatten())
            else:
                self.failed_samples.append(A_L.flatten())

    def get_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取采样结果"""
        verified_arr = np.array(self.verified_samples)
        failed_arr = np.array(self.failed_samples)
        return verified_arr, failed_arr

    def compute_covariance(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算采样后的协方差矩阵"""
        if not self.verified_samples or not self.failed_samples:
            raise ValueError("没有足够的采样区域")
        verified_arr = np.array(self.verified_samples)
        failed_arr = np.array(self.failed_samples)

        M_V = np.cov(verified_arr.T)
        M_F = np.cov(failed_arr.T)
        return M_V, M_F

    def reset(self):
        """重置采样器"""
        self.verified_samples.clear()
        self.failed_samples.clear()


class OptimizedCSR:
    """
    优化后的CSR方案，专门针对大规模子区域。
    """

    def __init__(self, d: int, use_incremental: bool = True, sample_ratio: float = None):
        """
        初始化优化后的CSR方案。

        Args:
            d: 状态空间维度
            use_incremental: 是否使用增量协方差计算
            sample_ratio: 采样比例（None表示不采样）
        """
        self.d = d
        self.use_incremental = use_incremental
        self.sample_ratio = sample_ratio

        if use_incremental:
            self.incremental_cov = IncrementalCovariance(d)

        if sample_ratio is not None:
            self.sampler = StatisticalSampler(d, sample_ratio)

        self.verified_A_L: List[np.ndarray] = []
        self.failed_A_L: List[np.ndarray] = []

    def process_region(self, A_L: np.ndarray, is_verified: bool):
        """
        处理一个区域。

        Args:
            A_L: 线性边界系数
            is_verified: 是否已验证
        """
        A_L = A_L.flatten()

        if is_verified:
            self.verified_A_L.append(A_L)
            if hasattr(self, 'incremental_cov'):
                self.incremental_cov.update_verified(A_L)
        else:
            self.failed_A_L.append(A_L)
            if hasattr(self, 'incremental_cov'):
                self.incremental_cov.update_failed(A_L)

        if hasattr(self, 'sampler'):
            self.sampler.add_region(A_L, is_verified)

    def analyze_subspace(self, k: Optional[int] = None, var_threshold: float = 0.9):
        """
        执行子空间分析。

        Args:
            k: 失败子空间维度（自动确定如果为None）
            var_threshold: 方差解释率阈值

        Returns:
            W_F: 失败子空间基（d x k）
            W_V: 已验证子空间基（d x (d-k)）
        """
        print("\n" + "-" * 60)
        print("CSR优化分析")
        print("-" * 60)

        # 计算协方差矩阵
        if self.sample_ratio is not None:
            print(f"使用统计采样 (比例={self.sample_ratio}):")
            print(f"  采样已验证区域: {len(self.sampler.verified_samples)}")
            print(f"  采样失败区域: {len(self.sampler.failed_samples)}")
            M_V, M_F = self.sampler.compute_covariance()
        elif self.use_incremental:
            print("使用增量协方差计算:")
            M_V, M_F = self.incremental_cov.get_covariance()
        else:
            print("使用完整协方差计算:")
            if not self.verified_A_L or not self.failed_A_L:
                raise ValueError("没有收集到区域数据")
            M_V = np.cov(np.array(self.verified_A_L).T)
            M_F = np.cov(np.array(self.failed_A_L).T)

        print(f"  已验证区域数量: {len(self.verified_A_L)}")
        print(f"  失败区域数量: {len(self.failed_A_L)}")

        print("\n协方差矩阵计算完成")

        # 广义特征值分解
        try:
            # 使用Cholesky分解求解
            M_V_reg = M_V + 1e-6 * np.eye(self.d)
            L_V = np.linalg.cholesky(M_V_reg)
            L_V_inv = np.linalg.inv(L_V)
            M_tilde = L_V_inv @ M_F @ L_V_inv.T
            eigenvalues, eigenvectors_tilde = np.linalg.eigh(M_tilde)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors_tilde = eigenvectors_tilde[:, idx]
            eigenvectors = L_V_inv.T @ eigenvectors_tilde

        except np.linalg.LinAlgError:
            print("警告: Cholesky分解失败，使用标准特征值分解")
            eigenvalues, eigenvectors = np.linalg.eigh(M_F)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        # 计算方差解释率
        total_var = np.sum(eigenvalues)
        var_explained = eigenvalues / total_var if total_var > 0 else eigenvalues
        cumulative_var = np.cumsum(var_explained)

        # 确定k
        if k is None:
            k = np.argmax(cumulative_var >= var_threshold) + 1
            k = min(k, self.d)

        W_F = eigenvectors[:, :k]
        W_V = eigenvectors[:, k:]

        print("\n子空间分析结果:")
        print(f"  失败子空间维度: {k}")
        print(f"  已验证子空间维度: {W_V.shape[1]}")
        print("\n特征值:")
        for i, (lam, var) in enumerate(zip(eigenvalues, var_explained)):
            print(f"  λ{i+1} = {lam:.4f} ({var*100:.1f}% 方差解释率)")

        print(f"\nTop-{k}个特征向量解释了 {cumulative_var[k-1]*100:.1f}% 的方差")

        if k == 1 and var_explained[0] > 0.6:
            print("\n✅ 成功: 单个方向捕获了大部分失败方差")
        elif cumulative_var[k-1] > 0.9:
            print("\n✅ 成功: 前k个方向捕获了90%以上的失败方差")
        else:
            print("\n⚠️ 警告: 可能需要更多方向")

        return W_F, W_V

    def reset(self):
        """重置CSR实例"""
        self.verified_A_L.clear()
        self.failed_A_L.clear()
        if hasattr(self, 'incremental_cov'):
            self.incremental_cov.reset()
        if hasattr(self, 'sampler'):
            self.sampler.reset()


class LBPCSRIntegration:
    """
    LBP验证流程与CSR方案的集成。
    """

    def __init__(self, d: int, use_incremental: bool = True, sample_ratio: float = None):
        """
        初始化集成类。

        Args:
            d: 状态空间维度
            use_incremental: 是否使用增量协方差计算
            sample_ratio: 采样比例
        """
        self.d = d
        self.csr = OptimizedCSR(d, use_incremental, sample_ratio)
        self.processed_regions = 0

    def hook_verification_region(self, region, is_verified, A_L):
        """
        在验证过程中钩子到每个区域的处理。

        Args:
            region: 区域对象（HyperrectangularRegion或SimplicialRegion）
            is_verified: 是否已验证
            A_L: 线性边界系数
        """
        self.csr.process_region(A_L, is_verified)
        self.processed_regions += 1
        if self.processed_regions % 10000 == 0:
            print(f"已处理 {self.processed_regions} 个区域...")

    def analyze(self, k: Optional[int] = None, var_threshold: float = 0.9):
        """
        在验证完成后执行CSR分析。

        Args:
            k: 失败子空间维度
            var_threshold: 方差解释率阈值

        Returns:
            W_F: 失败子空间基
            W_V: 已验证子空间基
        """
        print("\n" + "=" * 60)
        print(f"验证完成 - 执行CSR分析 (共 {self.processed_regions} 个区域)")
        print("=" * 60)
        return self.csr.analyze_subspace(k, var_threshold)

    def get_region_summary(self) -> Dict[str, Any]:
        """获取区域统计信息"""
        return {
            'total_regions': self.processed_regions,
            'verified_regions': len(self.csr.verified_A_L),
            'failed_regions': len(self.csr.failed_A_L),
            'verified_ratio': len(self.csr.verified_A_L) / self.processed_regions if self.processed_regions else 0
        }

    def reset(self):
        """重置集成类"""
        self.processed_regions = 0
        self.csr.reset()


# 测试和演示函数
def demo_large_scale_csr():
    """演示处理大规模区域的CSR方案"""
    d = 4  # 4维状态空间（如Cart-Pole）
    total_regions = 3000000  # 模拟300万个区域
    verified_ratio = 0.9  # 90%已验证

    # 创建集成类
    print("初始化CSR集成类...")
    csr_integration = LBPCSRIntegration(
        d=d, use_incremental=True, sample_ratio=0.01
    )

    print("模拟处理区域...")

    # 模拟已验证区域：围绕[1,0,1,0]的分布
    n_verified = int(total_regions * verified_ratio)
    center_verified = np.array([1, 0, 1, 0])
    for i in range(n_verified):
        if i % 100000 == 0:
            print(f"处理已验证区域: {i}/{n_verified}")
        A_L = center_verified + 0.1 * np.random.randn(d)
        csr_integration.hook_verification_region(
            region=None,
            is_verified=True,
            A_L=A_L
        )

    # 模拟失败区域：围绕[-0.5, 1, -0.5, 1]的分布
    n_failed = total_regions - n_verified
    center_failed = np.array([-0.5, 1, -0.5, 1])
    for i in range(n_failed):
        if i % 100000 == 0:
            print(f"处理失败区域: {i}/{n_failed}")
        A_L = center_failed + 0.3 * np.random.randn(d)
        csr_integration.hook_verification_region(
            region=None,
            is_verified=False,
            A_L=A_L
        )

    # 执行分析
    print("\n开始分析...")
    W_F, W_V = csr_integration.analyze()

    # 打印结果
    summary = csr_integration.get_region_summary()
    print("\n" + "=" * 60)
    print("CSR方案处理大规模区域演示")
    print("=" * 60)
    print(f"区域总数: {summary['total_regions']}")
    print(f"已验证区域: {summary['verified_regions']}")
    print(f"失败区域: {summary['failed_regions']}")
    print(f"验证率: {summary['verified_ratio']:.2%}")
    print(f"失败子空间维度: {W_F.shape[1]}")
    print(f"已验证子空间维度: {W_V.shape[1]}")

    print("\n✅ 演示完成！")
    print("CSR方案完全可以处理300万个区域")


if __name__ == "__main__":
    demo_large_scale_csr()
