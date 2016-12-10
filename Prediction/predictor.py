from collections import defaultdict
import math
import argparse
from binarypredictor import BinaryPredictor
import sys
from numpy import mean, std, min, max, median
import csv
import pickle

import gensim.models

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger=logging.getLogger(__name__)


class PDPS(BinaryPredictor):
    def __init__(self, filename, window=10, size=600, decay=5, balanced=False, prior=True,
                 dataset="ucsd", model="org"):
        self._window=window
        self._size=size
        self._decay=decay
        self._prior_pred=prior
        self._stopwordslist=[]
        self._dataset=dataset
        self._props={"window": window, "size": size, "decay": decay, "prior": prior,
                       "balanced": balanced, "dataset": dataset, "model": model}
        super(PDPS, self).__init__(filename)

    def dif_train(self, filename, workers):
        print(filename)
        self.base_train(filename, skipgram=1, workers=workers)

#    def train_5(self, filename):
#        print(filename)
#        self.base_train(filename, skipgram=1, mcount=5)

    def dropout_update(self, filename, dropout_round, keep_node_ratio):
        logger.info("start dropout update")
        print(filename)
        self.dropout_train(filename, dropout_round, keep_node_ratio)

    def update(self, filename, random_weight = True):
        ''' Update model with new data sets
            New data set specified by filename
        '''
        print("start naive update")
        print(filename)
        self.base_update(filename, random_weight)

    def save(self, filename):
        ''' Save the trained model in filename
        '''
        print(filename)
        self.base_save(filename)

    def load(self, filename):
        ''' Load the previously trained model from filename
        '''
        print(filename)
        self.base_load(filename)

    def predict(self, feed_events):
        te=len(feed_events)
        weighted_events=[(e,  math.exp(self._decay*(i-te+1)/te))
                           for i, e in enumerate(feed_events) if e in self._model.vocab]
        predictions=defaultdict(
            lambda: 1, {d: sim * (sim > 0) for d, sim in self._model.most_similar(
                weighted_events, topn=self._nevents)})
        return predictions

    def word_vector_graph(self, filename):
        from matplotlib import pyplot as plt
        self.counts=defaultdict(lambda: 0)
        with open(filename) as f:
            for s in f:
                sentense=s.split("|")[2].split(" ") + s.split("|")[3].replace("\n", "").split(" ")
                for e in sentense:
                    self.counts[e] += 1
        

        fig=plt.figure(figsize=(14, 14), dpi=180)
        colors={"d": "black", "p": "blue", "l": "red", "s": "green", "c": "orange"}
        plt.plot()
        ax=fig.add_subplot(111)

        events=[]
        for t in ["c", "s", "p", "d", "l"]:
            evs={e: c for e, c in self.counts.items() if e.startswith(t)}
            events += [x for x, y in sorted(evs.items(), key=lambda k: k[1], reverse=True)[:100]]

        for e in events:
            if e in ["c_V3000", "c_V053", "c_V502", "c_V3001", "c_290", "c_V290"]:
                continue
            if e in self._model.vocab:
                v=self._model[e]
                plt.plot(v[0], v[1])
                ax.annotate(e, xy=v, fontsize=10, color=colors[e[0]])

        p=ax.bar(0, [0], 0, color='blue')
        d=ax.bar(0, [0], 0, color='black')
        c=ax.bar(0, [0], 0, color='orange')
        s=ax.bar(0, [0], 0, color='green')
        l=ax.bar(0, [0], 0, color='red')
        plt.tick_params(axis='both', which='major', labelsize=16)
        ax.legend((d[0], p[0], l[0], c[0], s[0]), ('Diagnosis', 'Prescription', 'Lab test',
                                                   'Condition', 'Symptom'), loc=1, fontsize=22)
        plt.savefig('../Results/Plots/event_'+self._props["dataset"]+'_colored.png')
        sys.exit(0)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='SkipGram Similarity')
    parser.add_argument('-w', '--window', action="store", default=10, type=int,
                        help='Set max skip length between words (default: 10)')
    parser.add_argument('-s', '--size', action="store", default=600, type=int,
                        help='Set size of word vectors (default: 600)')
    parser.add_argument('-nw', '--nworkers', action="store", default=1, type=int,
                        help='Set number of workers (default: 1)')
    parser.add_argument('-t', '--task', action="store", default="CreateVocab", type=str,
                        help='Which operations to do "CreateVocab" or "MergeVocab" or "DNCEInit" or "DNCEUpdate" or "InitTrain" or "NaiveUpdate" or "CalcPTK" or "CalcStats"')
    parser.add_argument('-ifn', '--inputfilename', action="store", default="dataset1", type=str,
                        help="Which dataset should be used")
    parser.add_argument('-ofn1', '--outputfilename1', action="store", default="outputfile1", type=str,
                        help="Enter outputfile name1")
    parser.add_argument('-ofn2', '--outputfilename2', action="store", default="outputfile2", type=str,
                        help="Enter outputfile name2")
    parser.add_argument('-smn', '--savemodelname', action="store", default="savemodel", type=str,
                        help="Enter saved model name")
    parser.add_argument('-tmn', '--trainedmodelname', action="store", default="trainedmodel", type=str,
                        help="Enter trained model name")
    parser.add_argument('-gmn', '--goldmodelname', action="store", default="goldmodel", type=str,
                        help="Enter gold standard model name")
    parser.add_argument('-pp', '--privacyprotect', action="store", default=0, type=int,
                        help='Apply data privacy protection (0 for False, 1 for True) default 0')
    parser.add_argument('-cn', '--clusternumber', action="store", default=6, type=int,
                        help='How many clusters be used in privacy protection')
    parser.add_argument('-d', '--decay', action="store", default=5, type=float,
                        help='Set exponential decay through time (default: 5)')
    parser.add_argument('-p', '--prior', action="store", default=0, type=int,
                        help='Add prior probability (0 for False, 1 for True) default 0')
    parser.add_argument('-b', '--balanced', action="store", default=0, type=int,
                        help='Whether to use balanced or not blanaced datasets (0 or 1) default 0')
    parser.add_argument('-ds', '--dataset', action="store", default="mimic", type=str,
                        help='Which dataset to use "ucsd" or "mimic", default "mimic"')
    parser.add_argument('-m', '--model', action="store", default="org", type=str,
                        help='Which model to use "org" or "chao", default "org"')
    args=parser.parse_args()

    ds='mimic'

    data_path="../Data/mimic_seq/"
    if args.balanced:
        data_path="../Data/mimic_balanced/"

    prior=False if args.prior == 0 else True
    bal=False if args.balanced == 0 else True


    if args.task == "CreateVocab":
        model=PDPS(data_path + 'vocab', args.window, args.size, args.decay, bal, prior, ds,
                    args.model)

        model.output_vocab(data_path + args.inputfilename, data_path + args.outputfilename1, skipgram=1, workers = 1)

    if args.task == "MergeVocab":
        model=PDPS(data_path + 'vocab', args.window, args.size, args.decay, bal, prior, ds,
                    args.model)
        model._model = gensim.models.Word2Vec(sg=1, window=args.window, size=args.size, min_count=1, workers=args.nworkers)
        model._model.merge_update_vocab(merge_file1 = data_path + args.outputfilename1, merge_file2 = data_path + args.outputfilename2, 
                                       output_global = data_path + args.savemodelname, privacy_protect = args.privacyprotect,
                                       num_clust = args.clusternumber)

    if args.task == "DNCEInit":
        model=PDPS(data_path + 'vocab', args.window, args.size, args.decay, bal, prior, ds,
                    args.model)
        with open(data_path + args.savemodelname, 'rb') as input:
            model._model = pickle.load(input)
        exmp_count = model._model.corpus_count + model._model.corpus_count_1
        classic_alpha = model._model.alpha
        model._model.alpha_old = classic_alpha
        classic_minalpha = model._model.min_alpha
        model.continue_train_output(data_path + args.inputfilename, total_examples = exmp_count, readinput = False, outputindex = False)
        model._model.output_file(data_path + args.trainedmodelname)

    if args.task == "DNCEUpdate":
        model=PDPS(data_path + 'vocab', args.window, args.size, args.decay, bal, prior, ds,
                    args.model)
        with open(data_path + args.savemodelname, 'rb') as input:
            model._model = pickle.load(input)
        exmp_count = model._model.corpus_count + model._model.corpus_count_1
        classic_alpha = model._model.alpha
        classic_minalpha = model._model.min_alpha
        model._model.alpha = classic_alpha - (classic_alpha - classic_minalpha)* (1.0*model._model.corpus_count/exmp_count)
        model._model.alpha_old = classic_alpha
        model.continue_train_output(data_path + args.inputfilename, total_examples = exmp_count, readinput = False, outputindex = False)
        model._model.output_file(data_path + args.trainedmodelname)

    if args.task == "InitTrain":
        model=PDPS(data_path + 'vocab', args.window, args.size, args.decay, bal, prior, ds,
                    args.model)
        model.dif_train(data_path + args.inputfilename, workers = args.nworkers)
        model._model.output_file(data_path + args.trainedmodelname)

    if args.task == "NaiveUpdate":
        model=PDPS(data_path + 'vocab', args.window, args.size, args.decay, bal, prior, ds,
                    args.model)
        with open(data_path + args.savemodelname, 'rb') as input:
            model._model = pickle.load(input)
        model.update(data_path + args.inputfilename)
        model._model.output_file(data_path + args.trainedmodelname)

    if args.task == "DropoutUpdate":
        model=PDPS(data_path + 'vocab', args.window, args.size, args.decay, bal, prior, ds,
                    args.model)
        with open(data_path + args.savemodelname, 'rb') as input:
            model._model = pickle.load(input)
        model.dropout_update(data_path + args.inputfilename)
        model._model.output_file(data_path + args.trainedmodelname)

    if args.task == "CalcPTK":
        model=PDPS(data_path + 'vocab', args.window, args.size, args.decay, bal, prior, ds,
                    args.model)
        with open(data_path + args.savemodelname, 'rb') as input:
            model._model = pickle.load(input)
        model.report_top_k(golden_exist = True, input_golden = data_path + args.goldmodelname)
        print("average precision_top_k is " + str(mean(model.top_k_acc)))

    if args.task == "CalcStats":
        model=PDPS(data_path + 'vocab', args.window, args.size, args.decay, bal, prior, ds,
                    args.model)
        with open(data_path + args.savemodelname, 'rb') as input:
            model._model = pickle.load(input)
        model.valid(data_path + args.inputfilename)
        model.write_stats()   
        print("average AUC is " + str(mean(model.collect_auc)))    

    
