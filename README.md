# Bert-BiLSTM-CRF-pytorch
bert-bilstm-crf implemented in pytorch for named entity recognition.

```
python == 3.6
pytorch == 1.3
transformer == 2.1.1
```

### Data
* 首先将数据处理成`BIO`格式，processed文件夹下存放的是医疗命名实体识别的数据，代码可参考`data_process.ipynb`
* 下载中文BERT预训练模型,来自`pytorch-pretrained-bert`

### 模型训练
```
python main.py -- n_epochs 100 --finetuning --top_rnns
```


### 模型预测
```
python crf_predict.py
```


