# PhoBERT_Extraction
Project will extract vectors by sentences and words with one layer or concat more layers (not update)

#### You have to install VNCoreNLP, fairseq, PyTorch and download PhoBERT_base_fairseq or PhoBERT_large_fairseq and the state-of-art full paper as well as code here: https://github.com/VinAIResearch/PhoBERT

### Example usage PhoBERT Extraction

```python
from PhoBERTExtraction import LoadPretrainedModel,PhoBertExtraction,GenerateSuggest

MASTER_PATH="PhoBERT_large_fairseq"
BPE_PATH = "PhoBERT_large_fairseq/bpe.codes"
VNCORENLP_PATH="vncorenlp/VnCoreNLP-1.1.1.jar"

text = ["Thời gian gần đây, thông tin về việc vải thiều của Việt Nam đã được xuất khẩu sang Nhật và được bán với mức giá tốt khiến ai nấy đều rất vui mừng cho nền nông nghiệp nước nhà khi nông sản của chúng ta đã có thêm một bước tiến mới.",
        "So với mức giá của vải thiều được bán ở Việt Nam thì quả vải có giá 54k quả thật khiến nhiều người ngỡ ngàng. Tuy nhiên, nếu nhìn lại về quá trình trồng và tuyển chọn những quả vải đó, cho đến khi vận chuyển sang đến Nhật thì mức giá như trên có thể xem là khá hợp lý."]

load=LoadPretrainedModel(MODEL_PATH=MASTER_PATH,BPE_PATH=BPE_PATH,VNCORENLP_PATH=VNCORENLP_PATH)
phoBERT, rdrsegmenter=load.loadModel()
extract = PhoBertExtraction(phoBERT, rdrsegmenter)

tensorsSentence = extract.extraction_sentence_layer(index_layer=-1,lstSentences=text)
tensorsMultiSentence = extract.extraction_sentence_multilayers(num_bot_layers=4,lstSentences=text)
tensorsWord = extract.extraction_word_layer(index_layer=-1,lstSentences=text) 
print("Tensor Sentences:")
print(tensorsSentence.shape)
print("Tensor Multilayer Sentences:")
print(tensorsMultiSentence.shape)
print("Tensor Words:")
print([tensor.shape for tensor in tensorsWord])
```

```
loading archive file PhoBERT_large_fairseq
| dictionary: 64000 types
Tensor Sentences:
torch.Size([2, 1024])
Tensor Multilayer Sentences:
torch.Size([2, 4096])
Tensor Words:
[torch.Size([1, 44, 1024]), torch.Size([1, 25, 1024])]
```
### Example usage PhoBERT Suggest
```python
suggestText=['Hôm nay bầu trời thật đẹp']
suggest=GenerateSuggest(extract)
results=suggest.suggestExport(rawSentence=suggestText)
print('Top 20: ')
for result in results:
  print(result)
```

```
Nhìn trời thật đẹp 
Nhìn tôi tuyệt sáng 
Một gió thật đẹp 
Một mưa thật đẹp 
Một tôi thật đẹp 
Một trời mưa đẹp 
Ngắm sẽ nắng đẹp 
Khi mưa rất đẹp 
Có mưa rất xanh 
Cho mưa rất xanh 
Nhìn tôi mưa đẹp 
Một mưa rất xanh 
Nhìn tôi rất sáng 
Ngắm mưa rất xanh 
Vì thật tuyệt trong 
Vì tôi tuyệt trong 
Hôm_nay trời thật đẹp 
Ngắm trời thật đẹp 
Có tôi bạn cao 
Nhìn đã tuyệt trong 
```




