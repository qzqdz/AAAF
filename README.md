
requirement：
transformers                 4.23.0
torch                        1.13.1+cu116


running code：
python main.py --seed 3333  --data_path  /home/yindechun/Desktop/yanyu/data/ptance/Bernie_digital   --model_path  /home/yindechun/Desktop/yanyu/model/blank/twitter-roberta   --learning_rate  2e-5   --batch_size 8   --kg_each_seq_length 10  --kg_seq_length 48  --seq_length 128 --k 10  --epochs_num  5  --do_train True
