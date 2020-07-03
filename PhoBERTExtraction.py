from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE
from vncorenlp import VnCoreNLP
from fairseq import options  
from fairseq.models.roberta import RobertaModel
import argparse
import numpy as np
import torch
import copy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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

    def __init__(self,phoBERT,rdrsegmenter):

        self.phoBERT=phoBERT
        self.rdrsegmenter=rdrsegmenter

    def extraction_word_layer(self, index_layer, lstSentences):
        lstAll=[]
        sentences = [self.rdrsegmenter.tokenize(t) for t in lstSentences]
        for sentence in sentences:
          subwords = self.phoBERT.encode(" ".join(sentence[0]))
          tensorUnit=self.phoBERT.extract_features(subwords, return_all_hiddens=True)
          lstAll.append(tensorUnit[index_layer])
        return np.array(lstAll)
    
    def extraction_sentence_layer(self, index_layer, lstSentences):
        tensorAll=torch.empty(0,1024)
        lstAll=self.extraction_word_layer(index_layer,lstSentences)
        for tensorUnit in lstAll:
          tensorFilter=torch.from_numpy(np.mean(tensorUnit.detach().numpy()[0].T,axis=1).reshape(1,1024))
          tensorAll=torch.cat((tensorAll,tensorFilter))
        return tensorAll
    def extraction_sentence_multilayers(self, num_bot_layers, lstSentences):
        tensorAll=torch.empty(0,1024)
        lstLayers=[]
        for i in range(num_bot_layers):
            lstLayers.append(self.extraction_sentence_layer(index_layer=-1-i, lstSentences=lstSentences))
        tensorAll=torch.cat(tuple(lstLayers),1)
        return tensorAll
class GenerateSuggest:
    def __init__(self, extract):
        self.extract=extract

    def check(self, lstCount, lstMask, rawSentence):
      strCompare=''
      for k,z in enumerate(lstCount):
        strCompare+=lstMask[k][z]+' '
      tensorsSentence = self.extract.extraction_word_layer(index_layer=-1, lstSentences=[strCompare])[0].detach().numpy()[0]
      tensorsRaw = self.extract.extraction_word_layer(index_layer=-1,lstSentences=rawSentence)[0].detach().numpy()[0]
      return cosine_similarity(tensorsSentence, tensorsRaw)[0][0]

    def wordMask(self, text, num_top):
        words = self.extract.rdrsegmenter.tokenize(text[0])[0]
        lstMask=[]
        for i, token in enumerate(words):
            wordtest=copy.deepcopy(words)
            wordtest[i] = '<mask>'
            text_masked_tok = ' '.join(wordtest)
            topk_filled_outputs = self.extract.phoBERT.fill_mask(text_masked_tok, topk=num_top)
            lstMask.append([output[2] for output in topk_filled_outputs])
        return lstMask
    
    def getCosine(self, lstMask, rawSentence):
        dfThongke=pd.DataFrame(columns=['sentenceIndex','cosine'])
        for i in range(len(lstMask)):
          for k in range(10):
            lstCount=[k for c in range(len(lstMask))]
            lstCount[i]=0
            for j in range(10):
              dfThongke=dfThongke.append({
                                'sentenceIndex':';'.join(str(c) for c in lstCount),
                                'cosine': self.check(lstCount,lstMask,rawSentence)
              }, ignore_index=True)
              lstCount[i]+=1
        dfThongkeSort=dfThongke.sort_values(by=['cosine'],ascending=False)
        dfThongkeSort=dfThongkeSort.drop_duplicates(subset=None, keep='first', inplace=False)
        return dfThongkeSort

    def suggestExport(self, rawSentence, num_top=10, num_sentences=20):
        lstSuggest=[]
        lstMask=self.wordMask(rawSentence, num_top)
        dfThongke=self.getCosine(lstMask, rawSentence)
        for index in dfThongke.iloc[:num_sentences]['sentenceIndex']:
          lstIndex=[int(i) for i in index.split(';')]
          strExport=''
          for i,stt in enumerate(lstIndex):
            strExport+=lstMask[i][stt]+' '
          lstSuggest.append(strExport)
        return lstSuggest
            
    













    
