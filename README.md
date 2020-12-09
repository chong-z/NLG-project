# CS 269 NLG Project
_Generating Semi-Restricted Natural Language Adversarial Examples_

Based on https://github.com/timbmg/Sentence-VAE.

## Demo

We provide the interactive demo in Google Colab which is freely availible:
https://colab.research.google.com/github/chong-z/NLG-project/blob/master/CS269_NLG_demo.ipynb

## Command Lines

Here are some example commands if you prefer running it locally. Please follow the setup in the notebook `CS269_NLG_demo.ipynb`.

### Train a LSTM-based VAE
```
sh dowloaddata.sh
python train.py --data_dir data --epochs 10 --rnn_type lstm -tb
```

### Attack the model on HuggingFace
```
# Replace `models/sample-GRU/E9.pytorch` with your model.
python3 semi_attack.py -c models/sample-LSTM/E9.pytorch --rnn_type lstm --iter 5 --steps 10 --rseed 7 -v \
  --victim_model "distilbert-base-uncased-finetuned-sst-2-english" \
  --victim_sentence "a sometimes tedious film involving dull characters and boring motives ." \
  --reference_sentence "a deep and meaningful film that refreshes the soul ."
```

### Evaluation

We provide the script to reproduce the results in our report. The script attacks 100 examples from the validation split, and finds reference sentences from the training set automatically. For each attack it runs 20 iterations with 20 interpolations per examples, and report the best adversarial example it could find.

Note: The script may take hours to run depending on the GPU.

```
python3 semi_attack.py -c models/sample-LSTM/E9.pytorch --rnn_type lstm --iter 20 --steps 20 --rseed 3 --n_attacks 20 --ppl_use --n_eval 100 -v --victim_model "distilbert-base-uncased-finetuned-sst-2-english"
```
