from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE
from vncorenlp import VnCoreNLP
from fairseq import options  
from fairseq.models.roberta import RobertaModel
import argparse
import numpy as np
import torch

class LoadPretrainedModel:
    def __init__(self, MODEL_PATH,BPE_PATH,VNCORENLP_PATH):
        self.MODEL_PATH=MODEL_PATH
        self.BPE_PATH=BPE_PATH
        self.VNCORENLP_PATH=VNCORENLP_PATH
    def loadModel(self):
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE', default=self.BPE_PATH)
        args = parser.parse_args("")

        phoBERT = RobertaModel.from_pretrained(self.MODEL_PATH, checkpoint_file='model.pt')
        phoBERT.eval()
        phoBERT.bpe=fastBPE(args)

        rdrsegmenter = VnCoreNLP(self.VNCORENLP_PATH, annotators="wseg", max_heap_size='-Xmx500m')

        return phoBERT, rdrsegmenter
        
class PhoBertExtraction:

    def __init__(self, lstSentences,phoBERT,rdrsegmenter):

        self.phoBERT=phoBERT
        self.rdrsegmenter=rdrsegmenter
        self.lstSentences=lstSentences

    def extraction_word_layer(self, index_layer):
        lstAll=[]
        sentences = [self.rdrsegmenter.tokenize(t) for t in self.lstSentences]
        for sentence in sentences:
          subwords = self.phoBERT.encode(" ".join(sentence[0]))
          tensorUnit=self.phoBERT.extract_features(subwords, return_all_hiddens=True)
          lstAll.append(tensorUnit[index_layer])
        return np.array(lstAll)
    
    def extraction_sentence_layer(self, index_layer):
        tensorAll=torch.empty(0,1024)
        lstAll=self.extraction_word_layer(index_layer)
        for tensorUnit in lstAll:
          tensorFilter=torch.from_numpy(np.mean(tensorUnit.detach().numpy()[0].T,axis=1).reshape(1,1024))
          tensorAll=torch.cat((tensorAll,tensorFilter))
        return tensorAll
