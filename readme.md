

# 训练模型：
for sys in simple2d barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --train --activation Relu --hidden-sizes '32,64,32' --save-path '/data/mzm/mzm_Verification/verification-of-neural-cbf-mzm4/data/New_models_Hard_Relu_v1'; done


for sys in simple2d barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --train --activation Tanh --hidden-sizes '32,64,32' --save-path '/data/mzm/mzm_Verification/verification-of-neural-cbf-mzm4/data/New_models_Hard_Tanh_v1'; done


for sys in simple2d barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --train --activation Sigmoid --hidden-sizes '32,64,32' --save-path '/data/mzm/mzm_Verification/verification-of-neural-cbf-mzm4/data/New_models_Hard_Sigmoid_v1'; done



for sys in simple2d barr1 barr2 barr3 barr4; do   CUDA_VISIBLE_DEVICES=5,6  python3 experiments/barrier_certificate.py --system-type $sys --train --activation Relu --hidden-sizes '32,64,32' --save-path '/data/mzm/Repair_NCBF/data/New_models_Hard_Relu_v1'; done


for sys in simple2d barr1 barr2 barr3 barr4; do   CUDA_VISIBLE_DEVICES=5,6  python3 experiments/barrier_certificate.py --system-type $sys --train --activation Tanh --hidden-sizes '32,64,32' --save-path '/data/mzm/Repair_NCBF/data/New_models_Hard_Tanh_v1'; done


for sys in simple2d barr1 barr2 barr3 barr4; do   CUDA_VISIBLE_DEVICES=5,6  python3 experiments/barrier_certificate.py --system-type $sys --train --activation Sigmoid --hidden-sizes '32,64,32' --save-path '/data/mzm/Repair_NCBF/data/New_models_Hard_Sigmoid_v1'; done



for sys in simple2d barr1 barr2 barr3 barr4; do   CUDA_VISIBLE_DEVICES=7  python3 experiments/barrier_certificate.py --system-type $sys --train --activation LeakyReLU --hidden-sizes '32,64,32' --save-path '/data/mzm/Repair_NCBF/data/New_models_Hard_LeakyReLU_v1'; done





for sys in planarquad; do   CUDA_VISIBLE_DEVICES=7  python3 experiments/barrier_certificate.py --system-type $sys --train --activation LeakyReLU --hidden-sizes '32,64,32' --save-path '/data/mzm/Repair_NCBF/data/New_models_Hard_LeakyReLU_v1'; done


for sys in planarquad; do   CUDA_VISIBLE_DEVICES=7  python3 experiments/barrier_certificate.py --system-type $sys --train --activation Sigmoid --hidden-sizes '32,64,32' --save-path '/data/mzm/Repair_NCBF/data/New_models_Hard_Sigmoid_v1'; done


for sys in planarquad; do   CUDA_VISIBLE_DEVICES=7  python3 experiments/barrier_certificate.py --system-type $sys --train --activation Tanh --hidden-sizes '32,64,32' --save-path '/data/mzm/Repair_NCBF/data/New_models_Hard_Tanh_v1'; done




hiord2 hiord4 hiord6 hiord8 rendezvousdocking

for sys in rendezvousdocking; do     python3 experiments/barrier_certificate.py --system-type $sys --train --activation Tanh --hidden-sizes '32,64,32' --save-path '/data/mzm/mzm_Verification/verification-of-neural-cbf-mzm4/data/New_models_Hard_Tanh_v1'; done




# 验证

for sys in simple2d barr1 barr2 barr3 barr4 cartpole; do    CUDA_VISIBLE_DEVICES=7 python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Tanh --max-depth 12 --hidden-sizes '32,64,32'; done 

for sys in simple2d barr1 barr2 barr3 barr4 cartpole; do    CUDA_VISIBLE_DEVICES=7 python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Sigmoid --max-depth 12 --hidden-sizes '32,64,32'; done 

for sys in simple2d barr1 barr2 barr3 barr4 cartpole; do    CUDA_VISIBLE_DEVICES=7 python3 experiments/barrier_certificate.py --system-type $sys --verify --activation LeakyRelu --max-depth 12 --hidden-sizes '32,64,32'; done 



for sys in rendezvousdocking; do     python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Tanh --max-depth 15 --hidden-sizes '32,64,32'; done 





# 例如用 bash 循环

python3 New_repair/main_clean.py --activation Sigmoid --system barr3

python3 New_repair/main_v1.py --activation Sigmoid --system barr3


python3 New_repair/main_clean.py --activation Tanh --system simple_2d

for sys in simple_2d barr1 barr2 barr3; do
  for act in Relu Tanh Sigmoid; do
    python3 New_repair/main_v1.py --activation $act --system $sys
  done
done




python3 New_repair/main_v1.py --activation Tanh --system simple_2d
# 生成最终的表格
python3 New_repair/read_results.py






# 可视化区域
for sys in simple2d barr1 barr2 barr3; do     python visualize_regions.py --system "${sys}" --activation 'Tanh' --path "New_repair/regions/verified_regions_${sys}_Tanh.pt"; done


for sys in simple2d barr1 barr2 barr3; do     python visualize_regions.py --system "${sys}" --activation 'Relu' --path "New_repair/regions/verified_regions_${sys}_Relu.pt"; done

for sys in simple2d barr1 barr2 barr3; do     python visualize_regions.py --system "${sys}" --activation 'Sigmoid' --path "New_repair/regions/verified_regions_${sys}_Sigmoid_v1.pt"; done


for sys in simple2d barr1 barr2 barr3; do     python visualize_regions.py --system "${sys}" --activation 'LeakyReLU' --path "New_repair/regions/verified_regions_${sys}_LeakyReLU_v1.pt"; done





# 运行指定激活函数和系统
python3 New_repair/main.py --activation Tanh --system barr1
python3 New_repair/main.py --activation Tanh --system barr1
python3 New_repair/main.py -a Tanh -s simple_2d

python3 New_repair/main_v1.py --activation Sigmoid --system barr3
python3 New_repair/main_clean.py --activation Sigmoid --system barr3
python3 New_repair/main_v1.py --activation Tanh --system barr3

main_multi.py

# 一次性运行某个激活函数的全部5个系统
python3 New_repair/main_multi.py --activation Relu
python3 New_repair/main_multi.py -a Tanh




python3 New_repair/main.py --activation Sigmoid --system barr3


你帮我看一下 这段代码为什么会出现nan python3 New_repair/main_v1.py --activation Tanh --system barr3，而相同的代码，在relu激活函数下就没问题，这是为什么，帮我找到问题，当时先不要该我的代码。你可以写debug代码


main_v1.py 命令行参数

python New_repair/main_v1.py -a Tanh -s barr1 --iterations 5 --use-vmap
python New_repair/main_v1.py -a Relu -s barr1 --no-vmap       # 回退多线程
python New_repair/main_v1.py -a Tanh -s barr1 --batch-size 256 # 调整 batch
python New_repair/main_v1.py -a Tanh -s barr1 --lr 5e-4 --k-rank 200




python3 New_repair/main_clean.py --activation Sigmoid --system barr3



python3 New_repair/main_clean_v2.py --activation Tanh --system barr3 --iterations 10 --num_samples 500 --lr 1e-2

python3 New_repair/main_v2.py --activation Tanh   --system barr3 --iterations 10 --num_samples 500 --lr 1e-2




# main_clean_v8
for sys in barr1 barr2 barr3; do
  for act in Sigmoid; do
    python3 New_repair/main_clean_v8.py -a $act -s $sys --num-inner-steps 5 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done


for sys in barr1 barr2 barr3; do
  for act in Tanh; do
    python3 New_repair/main_clean_v8.py -a $act -s $sys --num-inner-steps 5 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done


for sys in barr1 barr2 barr3; do
  for act in Relu; do
    python3 New_repair/main_clean_v8.py -a $act -s $sys --num-inner-steps 5 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done




# main_v9
for sys in barr1 barr2 barr3; do
  for act in Sigmoid; do
    python3 New_repair/main_v9.py -a $act -s $sys --rs-n 50 --rs-sigma 0.001 --num-inner-steps 1 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done


for sys in barr1 barr2 barr3; do
  for act in Tanh; do
    python3 New_repair/main_v9.py -a $act -s $sys --rs-n 50 --rs-sigma 0.001 --num-inner-steps 1 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done



for sys in barr1 barr2 barr3; do
  for act in Relu; do
    python3 New_repair/main_v9.py -a $act -s $sys --rs-n 50 --rs-sigma 0.001 --num-inner-steps 1 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done




# main_clean_v8_ibp.py 


python3 New_repair/main_clean_v8_ibp.py -a Tanh -s barr1 --num-inner-steps 1 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5




# 初始验证
for sys in barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Tanh --max-depth 15 --hidden-sizes '32,64,32'; done 

for sys in barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Relu --max-depth 15 --hidden-sizes '32,64,32'; done 

for sys in barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Sigmoid --max-depth 15 --hidden-sizes '32,64,32'; done 



# main_clean_v9_ibp  IBP验证和算loss

python3 New_repair/main_clean_v9_ibp.py -a Tanh -s barr1 --num-inner-steps 1 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5


for sys in barr1; do   for act in Tanh; do     python visualize_regions.py --activation $act --system $sys       --path "New_repair/regions/verified_regions_${sys}_${act}_repaired_v9_ibp.pt"       --n "repaired_v9_ibp";   done; done




for sys in barr1 barr2 barr3 barr4; do
  for act in Sigmoid; do
    CUDA_VISIBLE_DEVICES=6,7 python3 New_repair/main_clean_v9_ibp.py -a $act -s $sys --num-inner-steps 1 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done


for sys in barr1 barr2 barr3 barr4; do
  for act in Relu; do
    python3 New_repair/main_clean_v9_ibp.py -a $act -s $sys --num-inner-steps 1 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done


for sys in barr1 barr2 barr3 barr4; do
  for act in Tanh; do
    python3 New_repair/main_clean_v9_ibp.py -a $act -s $sys --num-inner-steps 1 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done



for sys in barr1 barr2 barr3; do   for act in Relu Tanh Sigmoid; do     python visualize_regions.py --activation $act --system $sys       --path "New_repair/regions/verified_regions_${sys}_${act}_repaired_v9_ibp.pt"       --n "repaired_v9_ibp";   done; done


for sys in barr1 barr2 barr3; do   for act in Sigmoid; do     python visualize_regions.py --activation $act --system $sys       --path "New_repair/regions/verified_regions_${sys}_${act}_repaired_v9_ibp.pt"       --n "repaired_v9_ibp";   done; done

# main_clean_v10_ibp  LBP验证 IBP算loss

python3 New_repair/main_clean_v10_ibp.py -a Sigmoid -s barr1 --num-inner-steps 1 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5





for sys in barr4; do
  for act in Relu Tanh; do
    CUDA_VISIBLE_DEVICES=6,7 python3 New_repair/main_clean_v10_ibp.py -a $act -s $sys --num-inner-steps 1 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done


for sys in barr1 barr2 barr3 barr4; do
  for act in Sigmoid; do
    python3 New_repair/main_clean_v10_ibp.py -a $act -s $sys --num-inner-steps 1 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done


for sys in barr1 barr2 barr3; do
  for act in Tanh; do
    python3 New_repair/main_clean_v10_ibp.py -a $act -s $sys --num-inner-steps 1 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done



for sys in barr1 barr2 barr3; do   for act in Relu Tanh Sigmoid; do     python visualize_regions.py --activation $act --system $sys       --path "New_repair/regions/verified_regions_${sys}_${act}_repaired_v10_ibp.pt"       --n "repaired_v10_ibp";   done; done


for sys in barr1 barr2 barr3; do   for act in  Tanh; do     python visualize_regions.py --activation $act --system $sys       --path "New_repair/regions/verified_regions_${sys}_${act}_repaired_v10_ibp.pt"       --n "repaired_v10_ibp";   done; done



for sys in barr4; do
  for act in Sigmoid Relu Tanh; do
    CUDA_VISIBLE_DEVICES=6,7 python3 New_repair/main_clean_v10_ibp.py -a $act -s $sys --num-inner-steps 5 --lr 5e-3 --max-depth-start 10 --max-depth-limit 12 --depth-schedule "10,12" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done

# main_compare_v10

CUDA_VISIBLE_DEVICES=6,7 python New_repair/main_compare_v10.py -a Sigmoid -s barr1



simple2d barr1 barr2 barr3 barr4 cartpole



for sys in simple2d barr1 barr2 barr3 barr4 cartpole; do
  for act in Sigmoid LeakyRelu Tanh; do
    CUDA_VISIBLE_DEVICES=7 python3 New_repair/main_compare_v10.py -a $act -s $sys 
  done
done

for sys in  cartpole; do
  for act in Sigmoid Tanh; do
    CUDA_VISIBLE_DEVICES=7 python3 New_repair/main_compare_v10.py -a $act -s $sys 
  done
done

# Verify_LBP

CUDA_VISIBLE_DEVICES=6,7 python3 Verify_LBP.py -a Sigmoid -s barr2


for sys in barr3; do   for act in Relu Tanh Sigmoid; do     python visualize_regions.py --activation $act --system $sys       --path "New_repair/regions/verified_regions_${sys}_${act}_repaired_v9.pt"       --n "repaired_v9";   done; done


python visualize_regions.py --activation Relu --system barr3       --path "New_repair/nr_results_verify_lbp/lbp_regions_barr3_Relu_v9_ibp_maxdepth15.pt"  --n "repaired_v9_ibp_new"


python visualize_regions.py --activation Sigmoid --system barr2       --path "New_repair/nr_results_verify_lbp/lbp_regions_barr2_Sigmoid_v9_ibp_maxdepth15.pt"  --n "repaired_v9_ibp_new"



# main_clean_v10_lbp

 CUDA_VISIBLE_DEVICES=5,6 python New_repair/main_clean_v10_lbp.py -a Tanh -s barr1


# main_clean_v11_lbp

 for sys in  barr1 barr2 barr3; do
    CUDA_VISIBLE_DEVICES=7 python3 New_repair/main_clean_v11_lbp.py -a Relu -s $sys
done


for sys in barr4; do
  for act in Sigmoid Relu Tanh; do
    CUDA_VISIBLE_DEVICES=6,7 python3 New_repair/main_clean_v11_lbp.py -a $act -s $sys 
  done
done



for sys in simple2d barr1 barr2 barr3 barr4 cartpole; do
  for act in Sigmoid LeakyRelu Tanh; do
    CUDA_VISIBLE_DEVICES=7 python3 New_repair/main_clean_v11_lbp.py -a $act -s $sys 
  done
done



for sys in cartpole; do
  for act in Sigmoid LeakyRelu Tanh; do
    CUDA_VISIBLE_DEVICES=7 python3 New_repair/main_clean_v11_lbp.py -a $act -s $sys 
  done
done


# main_clean_v12_lbp_w


for sys in simple2d barr1 barr2 barr3 barr4 cartpole; do
  for act in Sigmoid LeakyRelu Tanh; do
    CUDA_VISIBLE_DEVICES=7 python3 New_repair/main_clean_v12_lbp_w.py -a $act -s $sys 
  done
done


# main_clean_v12_lbp_gp


for sys in simple2d barr1 barr2 barr3 barr4 cartpole; do
  for act in Sigmoid LeakyRelu Tanh; do
    CUDA_VISIBLE_DEVICES=7 python3 New_repair/main_clean_v12_lbp_gp.py -a $act -s $sys 
  done
done