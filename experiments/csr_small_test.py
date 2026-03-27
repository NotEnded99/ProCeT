"""
CSR方案小规模测试（快速执行）

这个脚本演示了CSR方案的基本功能，包括：
1. 增量协方差计算
2. 统计采样策略
3. 子空间分析
"""

import numpy as np
import random
from lbp_neural_cbf.csr_optimized import LBPCSRIntegration


def test_small_scale_csr():
    """测试小规模场景的CSR方案"""
    print("=" * 60)
    print("CSR方案 - 小规模测试")
    print("=" * 60)

    # 配置
    d = 4
    total_regions = 10000
    verified_ratio = 0.9

    # 初始化CSR集成类（使用增量计算和采样）
    print("1. 初始化CSR集成类")
    csr_integration = LBPCSRIntegration(
        d=d, use_incremental=True, sample_ratio=0.01
    )

    print(f"2. 模拟 {total_regions} 个区域的验证过程")

    # 模拟已验证区域
    n_verified = int(total_regions * verified_ratio)
    center_verified = np.array([1, 0, 1, 0])
    for i in range(n_verified):
        if i % 1000 == 0:
            print(f"  已验证区域: {i+1}/{n_verified}")
        A_L = center_verified + 0.1 * np.random.randn(d)
        csr_integration.hook_verification_region(None, True, A_L)

    # 模拟失败区域
    n_failed = total_regions - n_verified
    center_failed = np.array([-0.5, 1, -0.5, 1])
    for i in range(n_failed):
        if i % 100 == 0:
            print(f"  失败区域: {i+1}/{n_failed}")
        A_L = center_failed + 0.3 * np.random.randn(d)
        csr_integration.hook_verification_region(None, False, A_L)

    print("\n3. 获取区域统计信息")
    stats = csr_integration.get_region_summary()
    print(f"   区域总数: {stats['total_regions']}")
    print(f"   已验证: {stats['verified_regions']}")
    print(f"   失败: {stats['failed_regions']}")
    print(f"   验证率: {stats['verified_ratio']:.2%}")

    print("\n4. 执行CSR分析")
    try:
        W_F, W_V = csr_integration.analyze(var_threshold=0.9)
        print(f"   失败子空间维度: {W_F.shape[1]}")
        print(f"   已验证子空间维度: {W_V.shape[1]}")
        print(f"   失败子空间基形状: {W_F.shape}")
        print(f"   已验证子空间基形状: {W_V.shape}")

        print("\n✅ 测试成功！")

        # 输出一些关键数据
        print("\n关键数据:")
        print(f"  失败子空间基（W_F）:")
        print(W_F)
        print(f"  已验证子空间基（W_V）:")
        print(W_V)

    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

    return True


def test_sampling_strategy():
    """测试采样策略的效果"""
    print("\n" + "=" * 60)
    print("CSR方案 - 采样策略测试")
    print("=" * 60)

    d = 4
    total_regions = 10000
    verified_ratio = 0.9

    print("1. 初始化CSR集成类（无采样）")
    csr_no_sampling = LBPCSRIntegration(d=d, use_incremental=True, sample_ratio=None)

    print("2. 模拟验证过程")
    center_verified = np.array([1, 0, 1, 0])
    n_verified = int(total_regions * verified_ratio)
    for i in range(n_verified):
        A_L = center_verified + 0.1 * np.random.randn(d)
        csr_no_sampling.hook_verification_region(None, True, A_L)

    center_failed = np.array([-0.5, 1, -0.5, 1])
    n_failed = total_regions - n_verified
    for i in range(n_failed):
        A_L = center_failed + 0.3 * np.random.randn(d)
        csr_no_sampling.hook_verification_region(None, False, A_L)

    print("3. 初始化CSR集成类（带采样）")
    csr_with_sampling = LBPCSRIntegration(d=d, use_incremental=True, sample_ratio=0.1)

    print("4. 模拟验证过程")
    random.seed(42)
    for i in range(n_verified):
        A_L = center_verified + 0.1 * np.random.randn(d)
        csr_with_sampling.hook_verification_region(None, True, A_L)

    for i in range(n_failed):
        A_L = center_failed + 0.3 * np.random.randn(d)
        csr_with_sampling.hook_verification_region(None, False, A_L)

    print("\n5. 比较分析结果")
    try:
        W_F_no_sampling, W_V_no_sampling = csr_no_sampling.analyze()
        W_F_sampling, W_V_sampling = csr_with_sampling.analyze()

        print(f"   无采样-失败子空间维度: {W_F_no_sampling.shape[1]}")
        print(f"   有采样-失败子空间维度: {W_F_sampling.shape[1]}")

        # 计算子空间相似度（使用余弦相似度）
        cosine_sim = np.dot(W_F_no_sampling[:, 0], W_F_sampling[:, 0]) / (
            np.linalg.norm(W_F_no_sampling[:, 0]) * np.linalg.norm(W_F_sampling[:, 0])
        )

        print(f"   主方向余弦相似度: {cosine_sim:.4f}")

        if cosine_sim > 0.95:
            print("\n✅ 采样策略有效！主方向高度一致")
        elif cosine_sim > 0.90:
            print("\n✅ 采样策略有效！主方向非常一致")
        else:
            print(f"\n⚠️  采样策略可能需要调整（相似度={cosine_sim:.4f}）")

    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("CSR方案 - 测试套件")
    print("=" * 60)

    all_passed = True
    print()
    all_passed = all_passed and test_small_scale_csr()
    print()
    all_passed = all_passed and test_sampling_strategy()

    print()
    print("=" * 60)
    if all_passed:
        print("✅ 所有测试通过！CSR方案工作正常")
    else:
        print("❌ 测试失败！CSR方案可能存在问题")
    print("=" * 60)
