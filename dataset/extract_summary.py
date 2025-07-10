import re
import os
import yake
import json
import torch
import time
import logging
import shutil
# from fairseq.models.bart.hub_interface import BARTHubInterface
# from yake.highlight import TextHighlighter
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from utils import assign_label,write_jsonl,read_jsonl
from rake_nltk import Rake

logging.basicConfig(level=logging.DEBUG, format='%(message)s %(asctime)s')
logger = logging.getLogger(__name__)

# devices = ['cuda:0','cuda:0','cuda:1','cuda:1','cuda:2','cuda:2','cuda:3','cuda:3']
devices = [f'cuda:{i}' for i in range(8)]


def extract_summary_kwords_rake(samples,summary_type,device_name=None,do_sample=None):
    s = time.time()
    r = Rake()
    for i,sample in enumerate(samples):
        target_text = sample['target'].strip().lower().replace("<newline>"," ")
        kword_num = len(target_text.split())//40 + 1
        r.extract_keywords_from_text(target_text)
        kwords = r.get_ranked_phrases()[:kword_num]
        sample['summary'] = "<sep_1>".join(kwords)
        if i % 100 == 0:
            print(f'extract summary from {i}th sample')
            print(f"summary:{kwords}\ntarget:{target_text}")
    # write_jsonl(samples,f'./summary_extraction_temp/{os.getpid()}')
    print(f'finding keywords is completed in {time.time() - s},')
    return samples


def extract_summary_kwords(samples, summary_type,device_name=None,do_sample=None):
    s = time.time()
    pat = re.compile('(?<=<kw>).*?(?=<\/kw>)')
    extractor = yake.KeywordExtractor()
    # th = TextHighlighter(max_ngram_size=3)
    for i, sample in enumerate(samples):

        target_text = sample['target'].strip().lower()
        kwords = extractor.extract_keywords(target_text)

        # print(f"kwords:{[k for k, s in kwords]}")
        # print(f"target:{target_text}")

        def remove_duplicate(kwords):
            merge_kwords = []
            for kword, score in kwords:
                kword = kword.strip().lower()
                find = False
                for i, merge_kword in enumerate(merge_kwords):
                    if kword in merge_kword:
                        find = True
                        break
                    elif merge_kword in kword:
                        find = True
                        merge_kwords[i] = kword
                    else:
                        continue
                if not find:
                    merge_kwords.append(kword)
            return merge_kwords

        def merge_rank_kword(kwords, target_text):
            ordered_kwords = []
            for kword in kwords:
                mtc = re.search(kword, target_text)
                if mtc is not None:
                    pos = mtc.span()
                    overlap_kwords = []
                    update_ordered_kwords = []
                    for i, (or_word, or_pos) in enumerate(ordered_kwords):
                        if or_pos[0] > pos[1] or or_pos[1] < pos[0]:
                            update_ordered_kwords.append((or_word, or_pos))
                        else:
                            overlap_kwords.append((or_word, or_pos, i))

                    l = min([p[0] for w, p, i in overlap_kwords] + [pos[0]])
                    h = max([p[1] for w, p, i in overlap_kwords] + [pos[1]])
                    merge_kword = target_text[l:h]
                    update_ordered_kwords.append((merge_kword, (l, h)))
                    ordered_kwords = update_ordered_kwords
                    # print(f"ordered_kwords:{ordered_kwords}")
            ordered_kwords = sorted(ordered_kwords, key=lambda x: x[1][0])
            ordered_kwords = [k for k, p in ordered_kwords]
            return ordered_kwords

        # def find_kword():
        #    kwords = []
        #    hl_words = pat.findall(hl_text)
        #    for hl_word in hl_words:
        #        hl_word = hl_word.lower().strip()
        #        if hl_word not in kwords:
        #            kwords.append(hl_word)
        #    return kwords

        kwords = remove_duplicate(kwords)
        # print(f'remove duplicate:{kwords}')
        kwords = merge_rank_kword(kwords, target_text)
        sample['summary'] = ", ".join(kwords)
        # print(f"summary:{kwords}")
        if i % 100 == 0:
            print(f'extract summary from {i}th sample')
            print(f"summary:{kwords}")
    #write_jsonl(samples,f'./summary_extraction_temp/{os.getpid()}')
    print(f'finding keywords is completed in {time.time() - s},')
    return samples


def load_model(model_name, device_name, half=False):
    logger.info('==>>> loading {} model on device {}...'.format(model_name, device_name))
    # bart = torch.hub.load('pytorch/fairseq', model_name,hub_dir="/work/gk77/k77006/.cache",pretrained=True)
    # logger.info('args:{}'.format(vars(bart.args)))
    # bart = BARTHubInterface(bart.args, bart.task, bart.model)
    cache_dir = os.environ['CACHE_DIR'] if 'CACHE_DIR' in os.environ else '/work/gk77/k77006/.cache'
    bart = BartForConditionalGeneration.from_pretrained(model_name, # cache_dir="/home/u00483/.cache")
                                                        cache_dir=cache_dir)
    tokenizer = BartTokenizer.from_pretrained(model_name, # cache_dir= "/home/u00483/.cache")
                                              cache_dir=cache_dir)
    device = torch.device(device_name)
    bart.to(device)
    bart.eval()
    # if half:
    #    bart.half()
    return bart, tokenizer


def extract_summary_sentence(samples, summary_type, device_name,do_sample=False):
    if summary_type == 'abstract_novel':
        bart, tokenizer = load_model('facebook/bart-large',device_name)
        bart.load_state_dict(torch.load(novel_summary_model_path,map_location=device_name))
        logger.info(
            f"==>>> Loading model: novel_summarization_model...")
    elif summary_type == 'abstract_news':
        bart,tokenizer = load_model('facebook/bart-large-xsum',device_name)
        logger.info(
            f"==>>> Loading model: bart-large-xsum...")
    else:
        bart,tokenizer = load_model('facebook/bart-large-cnn',device_name)
        logger.info(
            f"==>>> Loading model: bart-large-cnn...")

    batches = [samples[i:i + bsz] for i in range(0, len(samples), bsz)]
    for batch in batches:
        batch_str = [sample['target'] for sample in batch]
        with torch.no_grad():
            inputs = tokenizer(batch_str, max_length=1024, return_tensors='pt', padding=True)
            max_length = 100#(assign_label(batch[0]['target_word_num']) + 1) * 10
            summaries = bart.generate(
                input_ids=inputs['input_ids'].to(device_name),
                attention_mask=inputs['attention_mask'].to(device_name),
                num_beams=1 if do_sample else 5,
                do_sample=do_sample,
                top_p=0.9 if do_sample else None,
                top_k=50 if do_sample else None,
                max_length=max_length,
                length_penalty=0.9,
                early_stopping=True)

        for idx in range(len(batch)):
            summary_ids = summaries[idx].view(-1).tolist()
            summary = tokenizer.decode(summary_ids, skip_special_tokens=True)
            batch[idx]['summary'] = summary
        logger.info(f'sour:{batch[0]["target"]}\nsummary:{batch[0]["summary"]}')
    #write_jsonl(samples,f'./summary_extraction_temp/{os.getpid()}')
    logger.info(f'finished extracted summary in sub process:{os.getpid()}')
    return samples



def read_file(file_path):
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            sample = json.loads(line)
            samples.append(sample)
    return samples


def extract_summary(sour_file, targ_file, summary_type, context, do_sample=False):
    #assert not os.path.exists('./summary_extraction_temp'), 'the temp folder for summary extraction exists'
    #os.makedirs('./summary_extraction_temp')
    logger.info(f"==>>> loading source file:{sour_file}")
    samples = read_file(sour_file)
    bsz = len(samples) // num_process + 1
    samples = [samples[i:i + bsz] for i in range(0, len(samples), bsz)]
    logger.info([len(samples[i]) for i in range(len(samples))])
    # pool = torch.multiprocessing.Pool(processes=num_process)
    pool = context.Pool(processes=num_process)
    results = []
    for i in range(num_process):
        result = pool.apply_async(func=extract_summary_kwords_rake if summary_type == "keywords" else extract_summary_sentence,
                                  args=[samples[i], summary_type, devices[i], do_sample])
        results.append(result)
        time.sleep(20)
    pool.close()
    pool.join()
    logger.info('finished extracting all summaries.')
    samples=[result.get() for result in results]
    samples = sum(samples,[])
    #samples=[]
    #for file_name in os.listdir('./summary_extraction_temp'):
    #    if os.path.isfile(os.path.join('./summary_extraction_temp',file_name)):
    #        samples += read_jsonl(os.path.join('./summary_extraction_temp',file_name))
    #shutil.rmtree('./summary_extraction_temp')
    with open(targ_file, 'w') as f:
        samples = [json.dumps(s) for s in samples]
        f.write('\n'.join(samples))
    logger.info(f'saving the data file to {targ_file}')


num_process = 1
bsz = 16

novel_summary_model_path = ('' if 'PROJECT_DIR' not in os.environ else os.environ['PROJECT_DIR']) + './../exp2/novel_summarization/exp3/best_model.pt'
#sets_size = {'train': 270000, 'valid': 1080, 'test': 1080}
sets_size = {'train': 360000, 'valid': 1440, 'test': 1440,'anno':1980}
#sets_size = {'train': 27000}
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dev_name', type=str)
    parser.add_argument('summary_type', type=str)
    parser.add_argument('config_name', type=str)
    parser.add_argument('--baseline',action="store_true")
    parser.add_argument('--sampling',action='store_true')
    args = parser.parse_args()
    logger.info(f'==>>> generate {args.summary_type} summary from {args.dev_name} set...')
    sour_file = f'./pg_data/{args.dev_name}/book_selected_pairs_{args.config_name}/{args.dev_name}{"_baseline_" if args.baseline else "_"}{sets_size[args.dev_name]}.jsonl'
    targ_dir = f'./pg_data/{args.dev_name}/book_train_{args.config_name}{"_sampling" if args.sampling else ""}'

    if not os.path.exists(targ_dir):
        os.makedirs(targ_dir)
    targ_file = os.path.join(targ_dir, f'{args.dev_name}_{args.summary_type}{"_baseline_" if args.baseline else "_"}{sets_size[args.dev_name]}.jsonl')
    context = torch.multiprocessing.get_context('spawn')
    extract_summary(sour_file, targ_file, args.summary_type,context,args.sampling)