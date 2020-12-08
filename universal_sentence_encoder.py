import torch
import tensorflow as tf
import tensorflow_hub as hub

class UniversalSentenceEncoder():
    def __init__(self):
        tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

        # Don't use all memory.
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        self.model = hub.load(tfhub_url)

    def cos_sim(self, s1, s2):
        e1, e2 = self.model([s1, s2]).numpy()

        e1 = torch.tensor(e1).to(torch.device("cuda:0"))
        e2 = torch.tensor(e2).to(torch.device("cuda:0"))


        e1 = torch.unsqueeze(e1, dim=0)
        e2 = torch.unsqueeze(e2, dim=0)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(e1, e2).tolist()[0]
