import json, argparse, sys, os,logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path,'./../dataset'))
from annotate_writing_mode import load_predict_model, read_jsonl, predict_test_set, write_jsonl

def anno_format_conversion(sour_file, target_file):
    with open(sour_file) as f:
        data = json.load(f)

    records =[]
    for inst in data:
        records.append({
            'target': inst['generated_paragraph'],
            'label': inst['mode'],
            'paragraph_index': inst['paragraph_index'],
            'idx':len(records)
        })

    with open(target_file, 'w') as f:
        records = '\n'.join([json.dumps(r) for r in records])
        f.write(records)


def annotate_book(pair_path, anno_pair_path, eval_path, trainer):
    logger.info(f'==>>> start to annotate pair file :{pair_path}')
    pairs = read_jsonl(pair_path)
    if len(pairs) == 0:
        return
    pred_labels, test_result = predict_test_set(trainer,pair_path,format='json')
    assert len(pred_labels) == len(pairs),'the pair number is inequivalent to that of predicted labels'
    for i in range(len(pairs)):
        pairs[i]['target_label'] = pred_labels[i]
    logger.info(f'==>>> save annotated pairs to file :{anno_pair_path}')
    write_jsonl(pairs,anno_pair_path)
    with open(eval_path,'w') as f:
        json.dump(str(test_result),f)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--eval_path', type=str, required=True)
    parser.add_argument('--proc_num', type=int, default=4)
    parser.add_argument('--model_checkpoint', type=str, required=True)
    args = parser.parse_args()

    result_dir = os.path.dirname(args.result_path)
    tmp_file = os.path.join(result_dir, 'tmp.json')
    anno_tmp_file = os.path.join(result_dir, 'tmp_anno.json')

    anno_format_conversion(args.result_path,tmp_file)
    trainer = load_predict_model(args.model_checkpoint)
    annotate_book(tmp_file, anno_tmp_file, args.eval_path, trainer)


    #ctxt = get_context('spawn')
    #annotate_book_corpus_multiprocess(
    #    book_pair_dir,
    #    anno_book_pair_dir,
    #    args.model_checkpoint,
    #    args.process_num,
    #    ctxt)
