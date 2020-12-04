# Try RNN
python train.py --data_dir data --epochs 10 --rnn_type rnn -tb --create_data
# Try our GRU
python train.py --data_dir data --epochs 10 --rnn_type gru -tb --create_data
# Try LSTM
python train.py --data_dir data --epochs 10 --rnn_type lstm -tb --create_data


# # Same experiments, but bidirectional
# # Try RNN
# python train.py --data_dir data --epochs 10 --rnn_type rnn -tb -bi
# # Try our GRU
# python train.py --data_dir data --epochs 10 --rnn_type gru -tb -bi
# # Try LSTM
# python train.py --data_dir data --epochs 10 --rnn_type lstm -tb -bi

# Same experiments, but more epochs and layers
# Try RNN
# python train.py --data_dir data --epochs 25 --rnn_type rnn --num_layers 3 -tb
# Try our GRU
# python train.py --data_dir data --epochs 25 --rnn_type gru --num_layers 3 -tb
# Try LSTM
# python train.py --data_dir data --epochs 10 --rnn_type lstm -tb -bi


# For inference
# python inference.py -c bin/lstm_init/E9.pytorch --rnn_type lstm
