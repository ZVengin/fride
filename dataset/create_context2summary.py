import json
from utils import read_jsonl,assign_label,write_jsonl,logger
from transformers import BartTokenizer
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sum2story_file',type=str)
    parser.add_argument('context2sum_file',type=str)
    parser.add_argument('--add_summary',action="store_true")
    parser.add_argument('--add_len',action="store_true")
    parser.add_argument('--add_mode',action='store_true')
    args = parser.parse_args()

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    logger.info(f'==>>> Loading data file from {args.sum2story_file}')
    samples = read_jsonl(args.sum2story_file)
    for sample in samples:
        target = sample['target']
        target_length_label = assign_label(len(tokenizer.tokenize(target)[:200]))
        summary = sample['summary']
        mode = sample['target_label']
        new_target = (f"{mode} <sep_0> " if args.add_mode else "")\
                     +f"{summary}"\
                     +(f" <sep_1> {target_length_label}" if args.add_len else "")
        sample['target'] = new_target
    write_jsonl(samples,args.context2sum_file)
    logger.info(f'==>>> saving data file to {args.context2sum_file}')