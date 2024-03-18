This repository provides data and codes used in the paper <em>A novel Transformer-based fully trainable point process</em> to Neuralcomputation.
# Contents
1. Instlation
2. Structure of folders and files
3. Citation
# Requirement
+ Python 3.7.13
+ Pytorch 1.11.0
# Installation
To get started with the project, follow these steps:
1. リポジトリをクローン.
```
$ git clone https://github.com/chihhar/RepVecMarkedPP.git
```

1.  requirements.txtから必要なパッケージをインストール:
```
$ pip install -r requirements.txt
```
1. データセットを [Google Drive]   (https://drive.google.com/drive/folders/1bDROZjdKLxUslbnUY7q0JbxQZhG-KSin?usp=drive_link) からダウンロードし、data フォルダないで解凍.

2. それぞれのデータセットを用いてモデルを訓練,評価するには、 Main.py を実行:
```
(training):$ python Main.py [-gene] [-imp] [-method] [--train]
(evaluation):$ python Main.py [-gene] [-imp] [-method] 
```
Option:
- train :
  - trainモード, 訓練時:True, 訓練済みモデルの検証:False, default:False
- gene
    - string: データセットの種類, default=h1
      - pinwheel トイデータ : pin3
      - seismic datasets (NCEDC) : jisin
      - SF police call datasets : 911_x_Address 
- 　trainvec_num:
  - int: seq_rep vecの数, default=3
- 　pooling_k:
  - int: anchor vecの数, default=3
- imp
    - string: contrastive lossの追加 (Utils.py を参照)
      - contrastive loss アリ:"all_contra"
      - contrastive loss ナシ："ncl"
- method :
  - string: モデルを指定 (Models.py 参照)
    - early-fusion : "early"
    - late-fusion : "late"
    - crossattention-fusion : "cross"
    - bottleneck-fusion : "btl"
    - 提案法 : "all"


# Structure of folders and files
```
.
├── kmeans.py
├── Main.py
├── plot_code.py
├── Utils.py
├── checkpoint
├── requirements.txt
├── data
│   ├── kaggle
│   │   └── police-department-incidents.csv
│   ├── toy_data_generater.py
│   ├── date_pickled.py
│   └── date_jisin.90016
├── dataloadFolder
│   ├── generate_pinwheel.py
│   └── set_data.py
├── log
├── pickled
├── plot
├── tmpplot
└── transformer
    ├── Constants.py
    ├── Layers.py
    ├── Models.py
    ├── Modules.py
    └── SubLayers.py

```
# データセットの指定
 set_data.py を参考にするとよい.以下は地震データセットを指定した例：
```python
(train): $ python Main.py -gene=jisin [-imp=""] [-method=""] --train
(valid): $ python Main.py -gene=jisin [-imp=""] [-method=""]
```
# [元論文] https://drive.google.com/file/d/1AhN0uN0AETXGPWedLo7r3XoJ23PK3b7o/view?usp=drive_link
