# # Try RNN
# python train.py --data_dir data --epochs 10 --rnn_type rnn -tb --create_data
# # Try our GRU
# python train.py --data_dir data --epochs 10 --rnn_type gru -tb --create_data
# # Try LSTM
# python train.py --data_dir data --epochs 10 --rnn_type lstm -tb --create_data


# For training with perplexity training
# python train.py --data_dir data --epochs 15 --rnn_type lstm -tb --perplexity_loss

# For inference
# python inference.py -c bin/lstm_init/E9.pytorch --rnn_type lstm

# For Adversarial attack
# python3 semi_attack.py -c bin/2020-Dec-05-22:30:44/E9.pytorch --iter 10 --steps 10 --victim_sentence "a sometimes tedious film involving dull characters and boring motives ." --reference_sentence "a deep and meaningful film that refreshes the soul ." --rseed 2
