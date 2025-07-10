import json
import nltk 
import re
import argparse
import numpy as np
from collections import defaultdict
from utils import logger
from partition_story import sent_len_clusters
from transformers import GPT2Tokenizer

def compute_score(l_tgt,l_hpy):
    assert len(l_tgt) == len(l_hpy)
    len_dict = defaultdict(list)
    for k,v in zip(l_tgt.tolist(),l_hpy.tolist()):
        len_dict[k].append(v)
    len_score = dict()
    for k,v in len_dict.items():
        len_score[k] = str(np.array(v).sum()/len(v))
    avg_score=str(abs(l_tgt-l_hpy).sum()/len(l_tgt))
    return len_score,avg_score

def extract_len_from_str(string):
    logger.info(f'prompt:{string}')
    p = re.compile(r'(?<=\[LEN\]) *(\d)+ *(?=\[SEP\])')
    r =p.search(string)
    logger.info(f'search result:{r}')
    if r is None:
        length = 0
    else:
        length = int(r.group(0).strip())
    return length
 
def count_len_of_text(tokenizer,text):
    sents = tokenizer.tokenize(text)
    length = len(sents)
    return length

def read_results(path):
    with open(path,'r') as f:
        results = []
        for line in f:
            line = json.loads(line.strip())
            results.append(line)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    pl_tgts = [int(r['test_paragraph_len']) for r in results]
    if sum(pl_tgts) == 0:
        pl_tgts = [int(r['target_paragraph_len']) for r in results ]
    pl_hpys = [count_len_of_text(tokenizer,r['hypothesis']) for r in results ]
    logger.info(f'target length:{pl_tgts}\nhypothesis length:{pl_hpys}')

    sl_tgts = [int(r['test_sent_len']) for r in results]
    if sum(sl_tgts) == 0:
        sl_tgts = [int(r['target_sent_len']) for r in results]
    #sl_hyps = [len(r['hypothesis'].split())/count_len_of_text(tokenizer,r['hypothesis'])
    #           for r in results]
    sl_hyps = []
    for r in results:
        if count_len_of_text(tokenizer,r['hypothesis']) !=0:
            sl_hyps.append(len(gpt_tokenizer.tokenize(r['hypothesis']))/count_len_of_text(tokenizer,r['hypothesis']) )
        else:
            logger.info(f'hypothesis is empty. {r}')
            sl_hyps.append(0)
    return (np.array(pl_tgts),np.array(pl_hpys)),(np.array(sl_tgts),np.array(sl_hyps))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_file',type=str)
    parser.add_argument('eval_file',type=str)
    args = parser.parse_args()

    logger.info('starting to test %s'%args.test_file)

    (pl_tgts,pl_hpys),(sl_tgts,sl_hyps) = read_results(args.test_file)
    plen_score,pavg_score = compute_score(pl_tgts,pl_hpys)
    slen_score,savg_score = compute_score(sl_tgts,sl_hyps)

    logger.info('saving result to %s'%args.eval_file)
    with open(args.eval_file,'w') as f:
        json.dump({'paragraph_len_avg_score':pavg_score,
                   'paragraph_len_score':plen_score,
                   'sent_len_avg_score':savg_score,
                   'sent_len_score':slen_score},f)



