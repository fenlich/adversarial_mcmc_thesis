
# current
python3 attack.py --attack_name charmer --max_steps 100 --beta 2 --lambda_lev 0.5 --tau 0 --device privateuseone:0 --loss margin --early_stop --lev_max 3 --dataset sst --model_name meta-llama/Llama-3.2-1B-Instruct --size 500 --skip_use
# charmer
python3 attack.py --attack_name charmer --tau 0 --device privateuseone:0 --loss margin --dataset sst --model_name meta-llama/Llama-3.2-1B-Instruct --size 500 --select_pos_mode batch --k 1
