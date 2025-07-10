import torch, json, os, spacy,random
import numpy as np
import pandas as pd
import time
from model import Model
from dataloader import get_dataloader
from config import *
from utils import logger,compute_Fscore,MODES
from dataset.utils import assign_label, has_utterance, check_describe_type,get_root
from collections import namedtuple
from sklearn.metrics import precision_recall_fscore_support
from writing_mode_classifier.finetune_classifier import predict_sample,load_predict_model
from utils import MODES,max_target
from collections import defaultdict

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Distinct-N/distinct_n')
from metrics import distinct_n_corpus_level

mode_classifier_path = os.environ['PROJECT_DIR']+"/exp/writing_mode_classifier/roberta-base-split3-1000-roberta_training/checkpoint-2000"
Candidate = namedtuple('Candidate',['index','context','summary','target_text','target_mode','target_length','generated_mode','generated_length','generated_text'])


def predict_text(model,
                 predict_file,
                 score_path,
                 group=False,
                 indent=False,
                 random_mode=True,
                 do_sample=True,
                 num_beams=1,
                 top_k=None,
                 temperature=1,
                 top_p=0.9,
                 num_return_sequences=1):
    logger.info(f"==>>> testing in {'random' if random_mode else 'target'} mode...")
    results = []
    sample_id = 0
    mode_trainer = load_predict_model(mode_classifier_path,out_dir=os.environ['PROJECT_DIR']+'/temp')
    for batch_id,batch in enumerate(model._dataset['test']):
        idxs,summaries,contexts,targets,lengths,modes = [],[],[],[],[],[]
        for sample in batch:
            target_token_ids = model._model.tokenizer.encode(sample.target, add_special_tokens=False)[:max_target]
            target_cluster_label = assign_label(len(target_token_ids))
            target = model._model.tokenizer.decode(target_token_ids)
            for mode in ([sample.target_label] if not random_mode else [
                m for m in MODES if m!=sample.target_label]):
                idxs.append(sample_id)
                contexts.append(sample.context)
                summaries.append(sample.summary)
                lengths.append(target_cluster_label)
                modes.append(mode)
                targets.append(target)

            sample_id +=1

        #if batch_id>10:
        #    break
        gen_texts = model._model.server_inference_batch(
            summaries=summaries,
            contexts=contexts,
            lengths=lengths,
            text_labels=modes,
            num_beams=num_beams,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k)
        #if quality:
        #    for sample in batch:
        #        sample_loss = model._model.get_loss([sample])
        #        sample_ppl = float(torch.exp(sample_loss).cpu().item())
        #        ppls.append(sample_ppl)
        for j, gen_text in enumerate(gen_texts):
            results.append(
                Candidate(
                    index=idxs[j],
                    context=contexts[j],
                    summary=summaries[j],
                    target_text=targets[j],
                    target_mode=modes[j],
                    target_length=len(model._model.tokenizer.tokenize(targets[j])),
                    generated_mode=predict_sample(mode_trainer.model,mode_trainer.tokenizer,gen_text),
                    generated_length=len(model._model.tokenizer.tokenize(gen_text)),
                    generated_text=gen_text
                )._asdict()
            )
        #logger.info(results[-1])

        #if sample_id>10:
        #    break

    group_results=defaultdict(list)
    if group:
        for result in results:
            group_results[result['index']].append(result)
        results = group_results

    with open(predict_file,'w') as f:
        json.dump(results,fp=f,indent=2 if indent else None)

    #precision, recall, Fscore = compute_mode_controllability(results)
    #mode_score=json.dumps({'precision': str(precision), 'recall': str(recall), 'F1': str(Fscore)})

    #if not quality:
        #distinct_score = compute_distinct_ngrams(results)
        #distinct_score = json.dumps(distinct_score)

    mode_recall = compute_mode_recall(results)
        #mode_recall = json.dumps({k:str(v) for k,v in mode_recall.items()})

    mean_distance, var_distance = compute_length_control(results, model._model.tokenizer)


    scores = pd.DataFrame(data={'dialogue_accuracy': [mode_recall['Dialogue']],
                                    'action_accuracy':[mode_recall['Action']],
                                    'description_accuracy':[mode_recall['Description']],
                                    'unknown_accuracy':[mode_recall['Unknown']],
                                    'distance_mean':[mean_distance],
                                    'distance_var':[var_distance]
                                    })
    #else:
        #mean_distance, var_distance = compute_length_control(results, model._model.tokenizer)
        #distance_score = json.dumps({'distance_mean':mean_distance,'distance_var':var_distance})
        #mode_recall = compute_mode_recall(results)
        #mode_recall = json.dumps({k: str(v) for k, v in mode_recall.items()})

        #scores = pd.DataFrame(data={#'perplexity': [str(ppls)],
        #                            'mode_controllability': [mode_score],
        #                            'length_controllability(distance)':[distance_score],
        #                            'mode_recall':[mode_recall]})

    logger.info('scores:\n{}'.format(scores))
    scores.to_csv(score_path,index=False)
    return scores


def compute_mode_controllability(results):
    predictions = []
    labels = []
    for r in results:
        predictions.append(r['generated_mode'])
        labels.append(r['target_mode'])
    p,r,f = compute_Fscore(predictions=predictions,labels=labels)
    return p,r,f


def compute_distinct_ngrams(results):
    grouped_results = defaultdict(list)
    for r in results:
        grouped_results[r['target_mode']].append(r['generated_text'].split())

    distinct_scores = dict()
    for mode in grouped_results.keys():
        ds1=distinct_n_corpus_level(grouped_results[mode],1)
        ds2=distinct_n_corpus_level(grouped_results[mode],2)
        distinct_scores[mode] = {'distinct_1':ds1,'distinct_2':ds2}
    return distinct_scores


def compute_length_control(results,tokenizer):
    distance = []
    for r in results:
        target_token_num = len(tokenizer.tokenize(r['target_text']))
        generated_token_num = len(tokenizer.tokenize(r['generated_text']))
        distance.append(abs(target_token_num-generated_token_num))
    mean_distance = np.array(distance).mean()
    var_distance = np.array(distance).var()
    return mean_distance,var_distance



def compute_mode_recall(results):
    grouped_results = defaultdict(list)
    for r in results:
        grouped_results[r['target_mode']].append(r['generated_mode'])
    precision_scores = defaultdict(list)
    for mode in grouped_results.keys():
        correct_modes = [predict_mode for predict_mode in grouped_results[mode]
                         if predict_mode == mode]
        precision_scores[mode] = len(correct_modes)/len(grouped_results[mode])
    return precision_scores

#from transformers import GPT2LMHeadModel, GPT2TokenizerFast
#from utils import batched_perplexity
#def compute_perplexity(results):
#    device = "cuda:1"
#    model_id = "gpt2-large"
#    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
#    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

#    stride = 512
#    batch_size = 4
#    test = {'text':[r['generated_text'] for r in results]}
#    ppl = batched_perplexity(model, test, tokenizer, batch_size, stride)
#    return ppl



def predict_text_interactive(model):
    retry = "no"
    while True:
        if retry == "no":
            context = input("the context of story:")
            if context == "exit":
                break
            summary = input("what do you want to generate:")
            mode = input("what writing mode do you want to use:")
            target_length = input("how many words do you expect to generate:")

        start_time = time.time()

        gen_text = model._model.server_inference(
            summary=summary,
            context=context,
            length=assign_label(int(target_length)),
            text_label=mode,
            num_beams=1,
            do_sample=True
        )
        logger.info(f'==>>>inferring in {time.time()-start_time} seconds.')
        gen_text_length = len(model._model.tokenizer.encode(gen_text[0], add_special_tokens=False))
        gen_text = gen_text[0].replace("<newline>", "\n")
        # logger.info(f"generated_text:{gen_text}")
        print(f'context:\n{context}\n\n' +
              f'summary:\n{summary}\n\n' +
              f'generation_length:{gen_text_length}, generation_mode:{mode}, generation:\n{gen_text}' + '\n' * 4)
        retry = input("do you want to retry? (yes/no)")


def main(config,
         finetuned_model_path,
         predict_text_path,
         score_path,
         is_interact=False,
         add_mode=False,
         add_length=False,
         add_summary=False,
         group=False,
         indent=False,
         random_mode=False):
    model = Model(config.model_name, config.model_path, cache_dir=cache_dir)
    model.add_special_tokens()
    model.set_input(add_mode=add_mode,add_length=add_length,add_summary=add_summary)
    model._model.load_state_dict(torch.load(finetuned_model_path,map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
    #model._model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.split_to_gpus(device_map=device_map[config.model_name])
    logger.info(f"model arguments:{vars(config)}...")
    if not is_interact:
        eval_loader = get_dataloader(config.test_file, config.batch_size, 2)
        model.load_data("test", eval_loader)
        predict_text(model, predict_text_path, score_path,group=group, indent=indent, random_mode=random_mode)
    else:
        predict_text_interactive(model)





import argparse
device_map={'bart':{'encoder':'cuda:0','decoder':'cuda:0'},
            'gpt2':{0:list(range(36))},
            't5':{0:list(range(24))}}
project_dir = os.environ["PROJECT_DIR"] if 'PROJECT_DIR' in os.environ else "/work/gk77/k77006/repos/summary2story_booksum"
data_dir = project_dir + "/dataset/pg_data/{}/book_train_{}"
model_paths = {"bart":"facebook/bart-large",
               "gpt2":"gpt2-large",
               "t5":"t5-large"}

cache_dir = os.environ['CACHE_DIR'] if 'CACHE_DIR' in os.environ else '/work/gk77/k77006/.cache'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_type", type=str, help="e.g, sentence or keywords")
    parser.add_argument("model_name", type=str, help="e.g. bart or gpt2 or t5")
    parser.add_argument("config_name", type=str, help="e.g, cfg4")
    parser.add_argument("exp_name", type=str, help="e.g, exp1")
    parser.add_argument("result_name",type=str,help="e.g, run1")
    parser.add_argument("--interact",action="store_true")
    parser.add_argument("--add_mode",action="store_true")
    parser.add_argument("--add_length",action="store_true")
    parser.add_argument("--add_summary",action="store_true")
    parser.add_argument("--group",action="store_true")
    parser.add_argument("--indent",action="store_true")
    parser.add_argument("--random_mode",action="store_true")

    args = parser.parse_args()
    cfg = TrainingArguments(
        train_file=None,#os.path.join(data_dir.format('train', args.config_name), f"train_{args.summary_type}_270000.jsonl"),
        validate_file=None,#os.path.join(data_dir.format('valid', args.config_name), f"valid_{args.summary_type}_1080.jsonl"),
        test_file=os.path.join(data_dir.format('test', args.config_name), f"test_{args.summary_type}_1440.jsonl"),
        model_name=args.model_name,
        model_path=model_paths[args.model_name],
        batch_size=8,
        epoch=None,
        lr=None,
        warmup_proportion=None,
        weight_decay=None,
        epsilon=None,
        train_device_map=None,
        eval_device_map=None,
        add_mode=args.add_mode,
        add_length=args.add_length,
        add_summary=args.add_summary,
        log_dir=f"./exp2/{args.model_name}_{args.summary_type}_{args.config_name}{'' if args.add_summary else '_nosum'}{'_mode' if args.add_mode else ''}{'_length' if args.add_length else ''}/{args.exp_name}"
    )

    finetuned_model_path = f"{cfg.log_dir}/best_model.pt"
    predict_text_path = f"{cfg.log_dir}/best_model_prediction_{args.result_name}"
    score_path = f"{cfg.log_dir}/best_model_score_{args.result_name}"
    main(cfg,
         finetuned_model_path,
         predict_text_path,
         score_path,
         is_interact=args.interact,
         add_mode=args.add_mode,
         add_length=args.add_length,
         add_summary=args.add_summary,
         group=args.group,
         indent=args.indent,
         random_mode=args.random_mode)
