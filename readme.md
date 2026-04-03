

# 训练模型：
for sys in simple2d barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --train --activation Relu --hidden-sizes '32,64,32' --save-path '/data/mzm/mzm_Verification/verification-of-neural-cbf-mzm4/data/New_models_Hard_Relu'; done

# 验证
for sys in simple2d barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Relu --hidden-sizes '32,64,32'; done


# 可视化区域
for sys in simple2d barr1 barr2 barr3; do     python visualize_regions.py --system "${sys}" --activation 'Tanh' --path "New_repair/regions/verified_regions_${sys}_Tanh.pt"; done


for sys in simple2d barr1 barr2 barr3; do     python visualize_regions.py --system "${sys}" --activation 'Relu' --path "New_repair/regions/verified_regions_${sys}_Relu.pt"; done

for sys in simple2d barr1 barr2 barr3; do     python visualize_regions.py --system "${sys}" --activation 'Sigmoid' --path "New_repair/regions/verified_regions_${sys}_Sigmoid.pt"; done

