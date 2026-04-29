"""
测试脚本：验证 main_compare_v10.py 修复后只有最后一层参数改变
直接加载原始模型和修复后模型进行对比
"""

import torch
import os

# 系统配置
activation = 'Sigmoid'
system_name = 'barr3'  # 可改成 barr2, barr3, barr4 测试其他系统

# 原始模型路径
model_dir = f"data/New_models_Hard_{activation}_v1"
original_path = f"{model_dir}/{system_name}_cbf.pth"

# 修复后模型路径
repaired_path = f"New_repair/regions/{system_name}_{activation}_cbf_repaired_compare_v10.pth"

print("=" * 70)
print("测试: Last Layer Repair 参数变化验证")
print("=" * 70)
print(f"原始模型: {original_path}")
print(f"修复后模型: {repaired_path}")
print("=" * 70)

# 加载模型
print("\n加载模型...")
state_orig = torch.load(original_path, map_location='cpu')
state_rep = torch.load(repaired_path, map_location='cpu')

print(f"\n原始模型参数 ({len(state_orig)}个):")
for name, p in state_orig.items():
    print(f"  {name}: {p.shape}")

print(f"\n修复后模型参数 ({len(state_rep)}个):")
for name, p in state_rep.items():
    print(f"  {name}: {p.shape}")

# 比较参数
print("\n" + "=" * 70)
print("参数对比结果")
print("=" * 70)
print(f"{'参数名':<20} {'原始值范围':<22} {'修复后值范围':<22} {'最大差异':<18}")
print("-" * 70)

changed = []
unchanged = []

for name in state_orig:
    orig_p = state_orig[name]
    rep_p = state_rep[name]

    orig_min, orig_max = orig_p.min().item(), orig_p.max().item()
    rep_min, rep_max = rep_p.min().item(), rep_p.max().item()
    max_diff = (orig_p - rep_p).abs().max().item()

    if max_diff > 1e-10:
        changed.append((name, max_diff))
        print(f"{name:<20} [{orig_min:.6f}, {orig_max:.6f}]   [{rep_min:.6f}, {rep_max:.6f}]   {max_diff:.6e}  **变化**")
    else:
        unchanged.append(name)
        print(f"{name:<20} [{orig_min:.6f}, {orig_max:.6f}]   [{rep_min:.6f}, {rep_max:.6f}]   {max_diff:.6e}")

print("-" * 70)

# 分析最后一层索引
print("\n" + "=" * 70)
print("验证结论")
print("=" * 70)

# 找出最后一层参数（通常是 network.6，因为有 4 个 Linear 层: 0, 2, 4, 6）
last_layer_params = [name for name in state_orig.keys() if '.6.' in name]
other_params = [name for name in state_orig.keys() if '.6.' not in name]

print(f"\n最后一层参数 (network.6.*): {last_layer_params}")
print(f"其他层参数: {other_params}")

changed_last = [name for name, _ in changed if '.6.' in name]
changed_other = [name for name, _ in changed if '.6.' not in name]

print(f"\n变化的最后一层参数: {changed_last}")
print(f"变化的其他层参数: {changed_other}")

if changed_other:
    print(f"\n❌ FAIL: 有 {len(changed_other)} 个其他层参数也发生了变化!")
    print(f"   变化的参数: {changed_other}")
else:
    print(f"\n✅ PASS: 只有最后一层参数发生变化，其他层参数完全不变")

if changed:
    print(f"\n变化详情:")
    for name, diff in changed:
        print(f"   {name}: max_diff={diff:.6e}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)