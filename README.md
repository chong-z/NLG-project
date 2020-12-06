# CS 269 NLG Project
_Generating Semi-Restricted Natural Language Adversarial Examples_

Based on https://github.com/timbmg/Sentence-VAE

## Train a GRU
```
sh dowloaddata.sh
python train.py --data_dir data --epochs 10 --rnn_type gru -tb --create_data
```

## Attack the model on HuggingFace
```
# Replace `models/sample-GRU/E9.pytorch` with your model.
python3 semi_attack.py -c models/sample-GRU/E9.pytorch --iter 5 --steps 10 --rseed 7 -v \
  --victim_model "distilbert-base-uncased-finetuned-sst-2-english" \
  --victim_sentence "a gorgeous , high-spirited musical from india that exquisitely mixed music , dance , song , and high drama ." \
  --reference_sentence "i finished my final exam at ucla"
```
