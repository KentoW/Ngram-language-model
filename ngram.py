# -*- coding: utf-8 -*-
# Ngram (vanilla, add, kneser-nay の3種類のスムージング, 最大4gram
# train     : 学習
# ppl       : パープレキシティ
# generate  : 生成 (best-N beam search, sampling beam search)
import sys
import argparse
import json
import numpy as np
import math
from collections import defaultdict
import time



def sample(preds, temperature=1.0):         # 勝手にノーマライズしてくれる便利サンプル
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class Ngram:
    def __init__(self, input_file, N=4, smoothing='kneser-nay'):
        if not (smoothing in ('kneser-nay', 'add', 'vanilla')):
            sys.stderr.write('Error : specify smoothing ("kneser-nay" or "add" or "vanilla")\n')
            exit(1)
        if N < 2 or 4 < N:
            sys.stderr.write('Error : specify 2 <= n <= 4\n')
            exit(1)
        self.voca = set()
        self.N = N
        self.V = 0
        self.smoothing = smoothing
        self.input_file = input_file
        self.total = sum(1 for line in open(self.input_file) if not line.startswith('# '))
        sys.stderr.write('===== Parameter =====\n')
        sys.stderr.write('Input file : %s\n'%self.input_file)
        sys.stderr.write('         N : %s\n'%self.N)
        sys.stderr.write(' Smoothing : %s\n'%self.smoothing)
        self.freqs = {}
        self.sums = {}
        for n in range(self.N-1):
            self.freqs[n+2] = defaultdict(lambda: defaultdict(float))   # N-gram FREQ (分子)
            self.sums[n+2] = defaultdict(float)                         # N-gram SUM  (分母)
        self.sets = defaultdict(lambda: set())                          # N-gram SET
        self.bi_num = defaultdict(set)                                  # uni_P_KNの分子，wordをいれたら直前wordのセットがかえってくる

    def train(self):
        sys.stderr.write('\n===== Training =====\n')
        count = 1
        for strm in open(self.input_file, 'r'):
            sys.stderr.write('\r[' + '='*(int(20*count/self.total)) + '>' + '-'*(20-int(20*count/self.total)) + '] (%06d/%06d)'%(count, self.total))
            if strm.startswith('#'):
                continue
            queue = ['<bos>', '<bos>', '<bos>']
            for word in strm.strip().split(" "):
                self.voca.add(word)
                queue.append(word)
                conditions = tuple(queue[-self.N:-1:])          # N-gram 確率（条件付き確率の条件部分）
                self.set_value(conditions, word)
            count += 1
        self.set_uni_p_kn()                                     # 最後にkneser-nay用のユニグラム確率を計算する
        self.V = len(self.voca)
        sys.stderr.write('\nVoca size : %s\n'%self.V)

    def set_value(self, conditions, v):                         # N-gram確率の計算に必要な値を計算する
        assert len(conditions) + 1 == self.N
        self.voca.add(v)
        if self.N == 2:
            condition = conditions[0]
            self.bi_num[v].add(condition)
            self.freqs[2][(condition, )][v] += 1
            self.sums[2][(condition, )] += 1
            self.sets[(condition, )].add(v)
        elif self.N == 3:
            condition_1 = conditions[0]
            condition_2 = conditions[1]
            self.bi_num[v].add(condition_2)
            self.freqs[2][(condition_2, )][v] += 1
            self.freqs[3][(condition_1, condition_2)][v] += 1
            self.sums[3][(condition_1, condition_2)] += 1
            self.sets[(condition_1, condition_2)].add(v)
        elif self.N == 4:
            condition_1 = conditions[0]
            condition_2 = conditions[1]
            condition_3 = conditions[2]
            self.bi_num[v].add(condition_3)
            self.freqs[2][(condition_3, )][v] += 1
            self.freqs[3][(condition_2, condition_3)][v] += 1
            self.freqs[4][(condition_1, condition_2, condition_3)][v] += 1
            self.sums[4][(condition_1, condition_2, condition_3)] += 1
            self.sets[(condition_1, condition_2, condition_3)].add(v)

    def set_uni_p_kn(self):
        self.voca = list(self.voca)
        freqs = [len(self.bi_num[v]) for v in self.voca]
        S = sum(freqs)
        self.uni_p_kn = dict(zip(self.voca, [float(freq) / S for freq in freqs]))

    def get_probs(self, conditions=['<bos>', '<bos>', '<bos>'], delta=0.5):
        self.voca = list(self.voca)
        conditions = list(conditions)
        if self.smoothing == 'kneser-nay':
            if conditions == []:
                return [self.uni_p_kn[word] for word in self.voca]
            else:
                if len(conditions) >= self.N:
                    conditions = conditions[-(self.N-1):]
                low_prob = self.get_probs(conditions[1:])
                N = len(conditions) + 1
                if tuple(conditions) not in self.freqs[N]: return low_prob
                g = 0.0 # for normalization
                freq = []
                for word in self.voca:
                    c = self.freqs[N][tuple(conditions)].get(word, 0)
                    if c > delta:
                        g += delta
                        c -= delta
                    freq.append(c)
                n = sum(self.freqs[N][tuple(conditions)].values())
                return [(c + g * lp) / n for c, lp in zip(freq, low_prob)]      # ここで0を返す時がある(これは正しい現象) 
        elif self.smoothing == 'add':
            if len(conditions) >= self.N:
                conditions = conditions[-(self.N-1):]
            return [(self.freqs[self.N][tuple(conditions)].get(word, 0)+delta)/(self.sums[self.N][tuple(conditions)]+delta*self.V) for word in self.voca]
        else:
            if len(conditions) >= self.N:
                conditions = conditions[-(self.N-1):]
            return [self.freqs[self.N][tuple(conditions)].get(word, 0)/self.sums[self.N][tuple(conditions)] for word in self.voca]

    def ppl(self, input_file, delta=0.5):
        sys.stderr.write('\n===== Perplexity =====\n')
        count = 1
        log_prob = 0.0
        total_words = 0.0
        word2idx = {v:int(i) for i, v in enumerate(self.voca)}
        for strm in open(input_file, 'r'):
            sys.stderr.write('\r[' + '='*(int(20*count/self.total)) + '>' + '-'*(20-int(20*count/self.total)) + '] (%06d/%06d)'%(count, self.total))
            if strm.startswith('#'):
                continue
            queue = ['<bos>', '<bos>', '<bos>']
            for word in strm.strip().split(" "):
                queue.append(word)
                conditions = tuple(queue[-self.N:-1:])
                probs = self.get_probs(conditions, delta)
                log_prob += math.log(probs[word2idx[word]], 2)
                total_words += 1
            count += 1
            #if count == 10: break                               # TODO For Debug
        neg_log_prob = -log_prob
        entropy = neg_log_prob / total_words
        ppl = math.pow(2, entropy)
        sys.stderr.write('\nPerplexity : %s\n'%ppl)
        return ppl

    def generate(self, window=5, mode='sample', temprature=1.0, max_word=20, delta=0.5):
        sys.stderr.write('\n===== Generation =====\n')
        word2idx = {v:int(i) for i, v in enumerate(self.voca)}
        idx2word = {int(i):v for i, v in enumerate(self.voca)}
        prob_forward = [[] for l in range(max_word)]       # prob_forward[t] = [(word_path, prob), ...]
        for t in range(max_word):       # 最大生成文長
            sys.stderr.write('\r[' + '='*(int(20*(t+1)/max_word)) + '>' + '-'*(20-int(20*(t+1)/max_word)) + '] (%03d/%03d)'%(t+1, max_word))
            if t == 0: 
                path = ('<bos>', '<bos>', '<bos>')
                probs = self.get_probs(path, delta)
                if mode == 'sample':
                    stack = set()
                    for s in range(10*window):
                        new_index = sample(probs, temprature)
                        new_word = idx2word[new_index]
                        new_path = path + (new_word, )
                        prob = math.log(probs[new_index])
                        stack.add((new_path, prob))
                        if len(stack) >= window:
                            break
                    for new in stack:
                        prob_forward[t].append(new)
                elif mode=='greedy':
                    for new_index in np.array(probs).argsort()[-window:][::-1]:
                        new_word = idx2word[new_index]
                        new_path = path + (new_word, )
                        prob = math.log(probs[new_index])
                        prob_forward[t].append((new_path, prob))
                else:
                    sys.stderr.write('Erorr: specify generation mode ("sample" or "greedy")\n')
                    exit(1)
            else:
                stack = set()
                for old_path, old_prob in prob_forward[t-1]:
                    conditions = old_path[-self.N+1::]
                    probs = self.get_probs(conditions, delta)
                    temp_stack = set()
                    if mode == 'sample':
                        for s in range(10*window):
                            new_index = sample(probs, temprature)
                            new_word = idx2word[new_index]
                            new_path = old_path + (new_word, )
                            new_prob = math.log(probs[new_index]) + old_prob
                            temp_stack.add((new_path, new_prob))
                            if len(temp_stack) >= window:
                                stack = stack.union(temp_stack)
                                break
                    elif mode=='greedy':
                        for new_index in np.array(probs).argsort()[-window:][::-1]:
                            new_word = idx2word[new_index]
                            new_path = old_path + (new_word, )
                            new_prob = math.log(probs[new_index]) + old_prob
                            temp_stack.add((new_path, new_prob))
                        stack = stack.union(temp_stack)
                for path, prob in sorted(list(stack), key=lambda x:x[1], reverse=True)[:window:]:
                    prob_forward[t].append((path, prob))
        sys.stderr.write('\n\n')
        return prob_forward[-1]


def main(args):
    ngram = Ngram(input_file=args.corpus, N=3, smoothing='kneser-nay')
    ngram.train()
    #ngram.ppl(input_file=args.corpus)               #ATTENTION!! very slow

    #np.random.seed(0)  # For Reproducibility
    generated_list = ngram.generate(window=5, mode='sample', temprature=1.1, max_word=30)

    print("生成結果")
    for words, log_prob in generated_list:
        print(log_prob, "".join(words[3::]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus", dest="corpus", default='./corpus.txt', type=str, help='specify corpus file')
    args = parser.parse_args()
    main(args)

