

# 训练模型：
for sys in simple2d barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --train --activation Relu --hidden-sizes '32,64,32' --save-path '/data/mzm/mzm_Verification/verification-of-neural-cbf-mzm4/data/New_models_Hard_Relu'; done

# 验证
for sys in simple2d barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Tanh --max-depth 8 --hidden-sizes '32,64,32'; done 

for sys in simple2d barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Sigmoid --max-depth 8 --hidden-sizes '32,64,32'; done 


for sys in simple2d barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Relu --max-depth 8 --hidden-sizes '32,64,32'; done 



python3 experiments/barrier_certificate.py --system-type simple2d --verify --activation Tanh --max-depth 8 --hidden-sizes '32,64,32'



# 例如用 bash 循环

python3 New_repair/main_clean.py --activation Sigmoid --system barr3
python3 New_repair/main_clean.py --activation Tanh --system simple_2d

for sys in simple_2d barr1 barr2 barr3 barr4; do
  for act in Relu Tanh Sigmoid; do
    python3 New_repair/main_clean.py --activation $act --system $sys
  done
done

# 生成最终的表格
python3 New_repair/read_results.py






# 可视化区域
for sys in simple2d barr1 barr2 barr3; do     python visualize_regions.py --system "${sys}" --activation 'Tanh' --path "New_repair/regions/verified_regions_${sys}_Tanh.pt"; done


for sys in simple2d barr1 barr2 barr3; do     python visualize_regions.py --system "${sys}" --activation 'Relu' --path "New_repair/regions/verified_regions_${sys}_Relu.pt"; done

for sys in simple2d barr1 barr2 barr3; do     python visualize_regions.py --system "${sys}" --activation 'Sigmoid' --path "New_repair/regions/verified_regions_${sys}_Sigmoid.pt"; done


python visualize_regions.py --system barr3 --activation 'Relu' --path "New_repair/regions/verified_regions_barr3_Relu_repaired.pt"

python visualize_regions.py --system barr3 --activation 'Sigmoid' --path "New_repair/regions/verified_regions_barr3_Sigmoid_repaired.pt"

python visualize_regions.py --system barr3 --activation 'Sigmoid' --path "New_repair/regions/verified_regions_barr3_Sigmoid_repaired2.pt"



python visualize_regions.py --system barr3 --activation 'Sigmoid' --path "New_repair/regions/verified_regions_barr3_Sigmoid_clean_repaired.pt" --n "clean_repaired"




# 运行指定激活函数和系统
python3 New_repair/main.py --activation Tanh --system barr1
python3 New_repair/main.py --activation Tanh --system barr1
python3 New_repair/main.py -a Tanh -s simple_2d

python3 New_repair/main_v1.py --activation Sigmoid --system barr3
python3 New_repair/main_clean.py --activation Sigmoid --system barr3

main_multi.py

# 一次性运行某个激活函数的全部5个系统
python3 New_repair/main_multi.py --activation Relu
python3 New_repair/main_multi.py -a Tanh




python3 New_repair/main.py --activation Sigmoid --system barr3





main_v1.py 命令行参数

python New_repair/main_v1.py -a Tanh -s barr1 --iterations 5 --use-vmap
python New_repair/main_v1.py -a Relu -s barr1 --no-vmap       # 回退多线程
python New_repair/main_v1.py -a Tanh -s barr1 --batch-size 256 # 调整 batch
python New_repair/main_v1.py -a Tanh -s barr1 --lr 5e-4 --k-rank 200




python3 New_repair/main_clean.py --activation Sigmoid --system barr3

