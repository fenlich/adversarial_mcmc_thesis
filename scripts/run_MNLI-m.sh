k=10
cd ..
#Bert
python3 attack.py --device cuda --loss margin --dataset mnli --model textattack/bert-base-uncased-MNLI --k $k --pga 0 --n_positions 20 --select_pos_mode batch --size 1000
python3 attack.py --device xpu --loss margin --dataset mnli --model textattack/bert-base-uncased-MNLI --k 10 --pga 0 --n_positions 20 --select_pos_mode batch --size 5 --skip_use

#Albert
python3 attack.py --device cuda --loss margin --dataset mnli --model Alireza1044/albert-base-v2-mnli --k $k --pga 0 --n_positions 20 --select_pos_mode batch --size 1000

#Roberta
python3 attack.py --device cuda --loss margin --dataset mnli --model textattack/roberta-base-MNLI --k $k --pga 0 --n_positions 20 --select_pos_mode batch --size 1000