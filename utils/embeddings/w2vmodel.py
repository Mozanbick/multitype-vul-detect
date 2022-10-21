import pickle
import os
from typing import List
from os import listdir
from os.path import join, isfile, exists
from utils.objects.cpg import Node
from utils.embeddings.embed_utils import extract_tokens
from gensim.models import Word2Vec
from configs import modelConfig as ModelConfig


class Corpus:

    def __init__(self, save_dir: str, batch_size=10000):
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.batch_idx = 0
        self.batch_count = 0
        self.corpus_set = []

    def __iter__(self):
        if self.corpus_set:
            self.save()
        for file in listdir(self.save_dir):
            file_path = join(self.save_dir, file)
            assert isfile(file_path)
            sample = pickle.load(open(file_path, "rb"))
            yield sample

    def add_corpus(self, nodes: List[Node], method):
        corpus = []

        for node in nodes:
            if not node.code:
                continue
            # tokenize
            tokens = extract_tokens(node.code, method)
            corpus += tokens

        self.corpus_set += corpus
        self.batch_count += 1
        # if reaching the max batch size, save to disk
        if self.batch_count >= self.batch_size:
            self.save()

    def save(self):
        if self.batch_count == 0:
            return
        self.batch_idx += 1
        if not exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_path = join(self.save_dir, f"batch_{ModelConfig.group}_{self.batch_idx}.pkl")
        f_corpus = open(save_path, "wb")
        pickle.dump(self.corpus_set, f_corpus, protocol=pickle.HIGHEST_PROTOCOL)
        f_corpus.close()
        self.corpus_set = []
        self.batch_count = 0

    def load(self, path: str):
        f_corpus = open(path, "rb")
        self.corpus_set = pickle.load(f_corpus)
        f_corpus.close()


def generate_w2vModel(
        corpus_path: str,
        w2v_model_path: str,
        size=30,
        alpha=0.001,
        window=5,
        min_count=1,
        min_alpha=0.0001,
        sg=0,
        hs=0,
        epoch=10,
        negative=10
):
    print("Training w2v model...")
    model = Word2Vec(sentences=Corpus(corpus_path), vector_size=size, alpha=alpha, window=window, min_count=min_count,
                     max_vocab_size=None, sample=0.001, seed=1, workers=2, min_alpha=min_alpha, sg=sg, hs=hs,
                     negative=negative, epochs=epoch)
    model.save(w2v_model_path)
    return model


def load_w2vModel(w2v_path: str):
    model = Word2Vec.load(w2v_path)
    return model
