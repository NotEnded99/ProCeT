

# 训练模型：
for sys in barr1 barr2 barr3 barr4 cartpole; do     python3 experiments/barrier_certificate.py --system-type $sys --train --activation Relu --hidden-sizes '32,64,32' --save-path '/data/mzm/mzm_Verification/verification-of-neural-cbf-mzm4/data/New_models_Hard_Relu_v1'; done


for sys in barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --train --activation Tanh --hidden-sizes '32,64,32' --save-path '/data/mzm/mzm_Verification/verification-of-neural-cbf-mzm4/data/New_models_Hard_Tanh_v1'; done


for sys in barr1 barr2 barr3 barr4 cartpole; do     python3 experiments/barrier_certificate.py --system-type $sys --train --activation Sigmoid --hidden-sizes '32,64,32' --save-path '/data/mzm/mzm_Verification/verification-of-neural-cbf-mzm4/data/New_models_Hard_Sigmoid_v1'; done



hiord2 hiord4 hiord6 hiord8 rendezvousdocking

for sys in rendezvousdocking; do     python3 experiments/barrier_certificate.py --system-type $sys --train --activation Tanh --hidden-sizes '32,64,32' --save-path '/data/mzm/mzm_Verification/verification-of-neural-cbf-mzm4/data/New_models_Hard_Tanh_v1'; done




# 验证
for sys in barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Tanh --max-depth 15 --hidden-sizes '32,64,32'; done 

for sys in barr1 barr2 barr3 barr4 cartpole; do     python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Sigmoid --max-depth 15 --hidden-sizes '32,64,32'; done 


for sys in barr1 barr2 barr3 barr4 cartpole; do     python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Relu --max-depth 15 --hidden-sizes '32,64,32'; done 



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

for sys in simple2d barr1 barr2 barr3; do     python visualize_regions.py --system "${sys}" --activation 'Sigmoid' --path "New_repair/regions/verified_regions_${sys}_Sigmoid.pt"; done


python visualize_regions.py --system barr3 --activation 'Relu' --path "New_repair/regions/verified_regions_barr3_Relu_repaired.pt"

python visualize_regions.py --system barr3 --activation 'Sigmoid' --path "New_repair/regions/verified_regions_barr3_Sigmoid_repaired.pt"

python visualize_regions.py --system barr3 --activation 'Sigmoid' --path "New_repair/regions/verified_regions_barr3_Sigmoid_repaired2.pt"



python visualize_regions.py --system barr3 --activation 'Sigmoid' --path "New_repair/regions/verified_regions_barr3_Sigmoid_clean_repaired.pt" --n "clean_repaired"


python visualize_regions.py --system barr3 --activation 'Tanh' --path "New_repair/regions/verified_regions_barr3_Tanh_repaired_v2.pt" --n "repaired_v2"


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









for sys in simple_2d barr1 barr2 barr3; do
  for act in Relu; do
    python3 New_repair/main_modified_v2.py --activation $act --system $sys  --iterations 10 --num_samples 100 --max_depth 12 --lr 5e-3 
  done
done

for sys in simple_2d barr1 barr2 barr3; do
  for act in Tanh; do
    python3 New_repair/main_modified_v2.py --activation $act --system $sys  --iterations 10 --num_samples 100 --max_depth 12 --lr 5e-3
  done
done

for sys in simple_2d barr1 barr2 barr3; do
  for act in Sigmoid; do
    python3 New_repair/main_modified_v2.py --activation $act --system $sys  --iterations 10 --num_samples 100 --max_depth 12 --lr 5e-3
  done
done




python visualize_regions.py --system barr1 --activation 'Sigmoid' --path "New_repair/regions/verified_regions_barr1_Sigmoid_clean_modified_v2.pt" --n "clean_repaired_v2"




for sys in simple2d barr1 barr2 barr3; do
  for act in Relu Tanh Sigmoid; do
    python visualize_regions.py --activation $act --system $sys \
      --path "New_repair/regions/verified_regions_${sys}_${act}_repaired_v6.pt" \
      --n "repaired_v6"
  done
done



for sys in simple2d ; do
  for act in Relu Tanh Sigmoid; do
    python visualize_regions.py --activation $act --system $sys \
      --path "New_repair/regions/verified_regions_simple_2d_${act}_repaired_v6.pt" \
      --n "repaired_v6"
  done
done






for sys in barr1 barr2 barr3; do
  for act in Relu; do
    python3 New_repair/main_v4.py --activation $act --system $sys 
  done
done

for sys in simple_2d barr1 barr2 barr3; do
  for act in Relu; do
    python3 New_repair/main_v1.py --activation $act --system $sys 
  done
done



python3 New_repair/main_v1.py --activation Relu --system barr3



for sys in simple_2d barr1 barr2 barr3; do
  for act in Relu; do
    python3 New_repair/main_v6.py -a $act -s $sys --rs-n 50 --rs-sigma 0.001 --num-inner-steps 5 --lr 5e-3
  done
done



for sys in simple_2d barr1 barr2 barr3; do
  for act in Sigmoid; do
    python3 New_repair/main_v6.py -a $act -s $sys --rs-n 50 --rs-sigma 0.001 --num-inner-steps 5 --lr 5e-3
  done
done


for sys in barr1 barr2 barr3; do
  for act in Tanh; do
    python3 New_repair/main_v6.py -a $act -s $sys --rs-n 50 --rs-sigma 0.001 --num-inner-steps 5 --lr 5e-3
  done
done



for sys in simple_2d barr1 barr2 barr3; do
  for act in Relu; do
    CUDA_VISIBLE_DEVICES=0 python3 New_repair/main_v6.py -a $act -s $sys --rs-n 50 --rs-sigma 0.001 --num-inner-steps 5 --lr 5e-3 --max-depth 15
  done
done



python New_repair/main_v5.py \
    --activation Tanh \
    --system barr1 \
    --rs-n 50 \
    --rs-sigma 0.001 \
    --num-inner-steps 5 \
    --lr 5e-3 \
    --max-depth 12





for sys in simple_2d barr1 barr2 barr3; do
  for act in Relu; do
    python3 New_repair/main_v7.py -a $act -s $sys --rs-n 50 --rs-sigma 0.001 --num-inner-steps 5 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done


# 


for sys in simple_2d barr1 barr2 barr3; do
  for act in Sigmoid; do
    python3 New_repair/main_v8.py -a $act -s $sys --rs-n 50 --rs-sigma 0.001 --num-inner-steps 5 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done


for sys in simple_2d barr1 barr2 barr3; do
  for act in Relu; do
    python3 New_repair/main_v8.py -a $act -s $sys --rs-n 50 --rs-sigma 0.001 --num-inner-steps 5 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done


for sys in simple_2d barr1 barr2 barr3; do
  for act in Tanh; do
    python3 New_repair/main_v8.py -a $act -s $sys --rs-n 50 --rs-sigma 0.001 --num-inner-steps 5 --lr 5e-3 --max-depth-start 10 --max-depth-limit 15 --depth-schedule "10,12,15" --plateau-threshold 0.5 --max-stagnant-iterations 5
  done
done



 # 画图

for sys in simple2d ; do   for act in Relu Tanh Sigmoid; do     python visualize_regions.py --activation $act --system $sys       --path "New_repair/regions/verified_regions_simple_2d_${act}_repaired_v7.pt"       --n "repaired_v7";   done; done

for sys in simple2d barr1 barr2 barr3; do   for act in Relu Tanh Sigmoid; do     python visualize_regions.py --activation $act --system $sys       --path "New_repair/regions/verified_regions_${sys}_${act}_repaired_v7.pt"       --n "repaired_v7";   done; done



for sys in barr3; do   for act in Relu Tanh Sigmoid; do     python visualize_regions.py --activation $act --system $sys       --path "New_repair/regions/verified_regions_${sys}_${act}_repaired_v9.pt"       --n "repaired_v9";   done; done


for sys in simple2d ; do   for act in Relu Tanh Sigmoid; do     python visualize_regions.py --activation $act --system $sys       --path "New_repair/regions/verified_regions_simple_2d_${act}_repaired_v9.pt"       --n "repaired_v9";   done; done



for sys in barr1 barr2 barr3 ; do   for act in Relu Tanh Sigmoid; do     python visualize_regions.py --activation $act --system $sys       --path "New_repair/regions/verified_regions_${sys}_${act}_v1.pt"       --n "None";   done; done



for sys in barr1 barr2 barr3; do   for act in Relu Tanh Sigmoid; do     python visualize_regions.py --activation $act --system $sys       --path "New_repair/regions/verified_regions_${sys}_${act}_repaired_v8_clean.pt"       --n "repaired_v8_clean";   done; done







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


for sys in barr1; do   for act in Tanh; do     python visualize_regions.py --activation $act --system $sys       --path "New_repair/regions/verified_regions_${sys}_${act}_repaired_v8_ibp.pt"       --n "repaired_v8_ibp";   done; done


# 初始验证
for sys in barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Tanh --max-depth 15 --hidden-sizes '32,64,32'; done 

for sys in barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Relu --max-depth 15 --hidden-sizes '32,64,32'; done 

for sys in barr1 barr2 barr3 barr4; do     python3 experiments/barrier_certificate.py --system-type $sys --verify --activation Sigmoid --max-depth 15 --hidden-sizes '32,64,32'; done 
