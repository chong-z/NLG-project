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
