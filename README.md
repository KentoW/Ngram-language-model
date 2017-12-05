# Ngram-language-model
## 概要
N-gram言語モデルをPythonで実装  
加算スムージングとKneser-nayスムージングを実装
## ngram.pyの使い方
```python
# Sample code.
from ngram import Ngram

ngram = Ngram(input_file='corpus.txt', N=3, smoothing='kneser-nay')
ngram.train()
ngram.ppl(input_file=args.corpus)               #ATTENTION!! very slow
generated_list = ngram.generate(window=5, mode='sample', temprature=1.1, max_word=30, delta=0.5)

print("生成結果")
for words, log_prob in generated_list:
    print(log_prob, "".join(words[3::]))
```
## パラメータ
input_file : 入力データ  
N : N-gramのN  
smoothing : スムージングの種類（vanilla, add, kneser-nay）  
window : ビーム探索の探索窓幅  
mode : ビーム探索の方法（greedy, sample）  
temprature : ビーム探索の方法がsampleのとき、確率分布から単語をsampleする温度  
max_word : 生成する単語数  
delta: 加算スムージング及びKneser-nayスムージングのパラメータ

## 入力データ
1単語をスペースで分割した1行1文
先頭に#(シャープ)記号を入れてコメントアウトを記述可能
```
# 文1
単語1 単語2 単語3 ...
# 文2
単語10 単語11 単語10 ...
...
```
例として[Wiki.py](https://github.com/KentoW/wiki)を使用して収集した アニメのあらすじ文章をcorpus.txtに保存
