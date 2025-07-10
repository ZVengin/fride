import argparse,logging,json
from eva.bleu import BLEU
import json
import os
import copy
import numpy as np
import pandas as pd
import sys

from abc import abstractmethod


_DESCRIPTION = """\
FwPPL, or forward perplexity, is a metric which measure the language model score.
"""

_KWARGS_DESCRIPTION = """
FwPPL score.
Args:
`data`： (list of dict including reference and candidate).
`model_id`: refer to `https://huggingface.co/models` for all available models.
`model_name_or_path`: can be the same with model_id or a path of checkpoint.
Returns:
    `res`: dict of list of scores.
"""

"""
Copied from nltk.ngrams().
"""
from itertools import chain


class Metrics:
    def __init__(self):
        self.name = 'Metric'

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def get_score(self):
        pass



def test_metric(data):
    from eva.tokenizer import SimpleTokenizer, PretrainedTokenizer
    tokenizer = SimpleTokenizer(method="nltk")
    union_path = "/home/u00483/repos/PhD-Project/summary2story_booksum/Union"
    eval_path="/home/u00483/repos/PhD-Project/summary2story_booksum/hyper_search_trial"

    metric_score = {}

    from eva.bleu import BLEU
    bleu_metric = BLEU(tokenizer=SimpleTokenizer(method="nltk"))
    # print(bleu_metric.info())
    metric_score.update(bleu_metric.compute(data))

    #from eva.meteor import METEOR
    #meteor_metric = METEOR()
    # print(meteor_metric.info())
    #metric_score.update(meteor_metric.compute(data))

    from eva.Rouge import ROUGE
    rouge_metric = ROUGE()
    # print(rouge_metric.info())
    metric_score.update(rouge_metric.compute(data))

    from eva.fwppl import FwPPL as EvalPPL
    fwppl_metric = EvalPPL(model_id="gpt2-large", model_name_or_path="gpt2-large")
    #print(fwppl_metric.info())
    metric_score.update(fwppl_metric.compute(data))
    #ft_fwppl_metric = FwPPL(model_id="gpt2", model_name_or_path="your_model_path")
    #print(ft_fwppl_metric.info())
    #metric_score.update(ft_fwppl_metric.compute(data))

    from eva.bertscore import BERTScore
    bertscore_metric = BERTScore(model_type="bert-base-uncased")
    # print(bertscore_metric.info())
    metric_score.update(bertscore_metric.compute(data))


    for metric,scores in metric_score.items():
        metric_score[metric] = [np.array(scores).mean()]
    return metric_score

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_file",type=str)
    parser.add_argument("score_file",type=str)
    args = parser.parse_args()

    results, test_samples = [],[]
    with open(args.prediction_file,'r') as f:
        data = json.load(f)
        for line in data:
            line['reference'] = [line.pop('target_text')]
            line['candidate'] = line.pop('generated_text')
            results.append(line)

    metric_score = test_metric(results)
    metric_score_df = pd.DataFrame(data=metric_score)
    metric_score_df.to_csv(args.score_file,index=False)

