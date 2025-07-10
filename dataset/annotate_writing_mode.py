import os,sys,torch
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir,'./../writing_mode_classifier'))
from torch.multiprocessing import Pool,get_context

from utils import read_jsonl,write_jsonl,logger
from finetune_classifier import load_predict_model,predict_test_set
def annotate_book(pair_path, anno_pair_path,trainer):
    logger.info(f'==>>> start to annotate pair file :{pair_path}')
    pairs = read_jsonl(pair_path)
    if len(pairs) == 0:
        return
    pred_labels = predict_test_set(trainer,pair_path,format='json')
    assert len(pred_labels) == len(pairs),'the pair number is inequivalent to that of predicted labels'
    for i in range(len(pairs)):
        pairs[i]['target_label'] = pred_labels[i]
    logger.info(f'==>>> save annotated pairs to file :{anno_pair_path}')
    write_jsonl(pairs,anno_pair_path)


def annotate_book_group(book_pair_dir,anno_book_pair_dir,model_checkpoint, file_names,gpu_id):
    #torch.cuda.set_device(gpu_id)
    trainer = load_predict_model(model_checkpoint)
    for idx, file_name in enumerate(file_names):
        pair_path = os.path.join(book_pair_dir, file_name)
        anno_pair_path = os.path.join(anno_book_pair_dir, file_name)
        annotate_book(pair_path, anno_pair_path, trainer)
        if idx % 100 == 0:
            logger.info(f'==>>> annotated [{idx}]/[{len(file_names)}] pair files.')

def annotate_book_corpus_multiprocess(book_pair_dir,anno_book_pair_dir,model_checkpoint,process_num,context):
    file_names = [file_name for file_name in os.listdir(book_pair_dir)
                  if os.path.isfile(os.path.join(book_pair_dir, file_name))]
    anno_file_names = [file_name for file_name in os.listdir(anno_book_pair_dir)
                  if os.path.isfile(os.path.join(anno_book_pair_dir, file_name))]
    file_names = list(set(file_names)-set(anno_file_names))
    logger.info(f'there are {len(anno_file_names)} books having been annotated, and {len(file_names)} books need to '
                f'be annotated.')
    group_size = len(file_names)//process_num+1
    groups = [file_names[i:i+group_size] for i in range(0,len(file_names),group_size)]
    p = context.Pool(process_num)
    procs = []
    for i in range(len(groups)):
        logger.info(f'==>>> starting the {i}th process...')
        proc=p.apply_async(func=annotate_book_group,args=[book_pair_dir,
                                                     anno_book_pair_dir,
                                                     model_checkpoint,
                                                     groups[i],
                                                     i])
        procs.append(proc)
        logger.info(f'==>>> ending the {i}th process...')
    p.close()
    p.join()
    outs = [proc.get() for proc in procs]




def annotate_book_corpus(book_pair_dir,anno_book_pair_dir,model_checkpoint):
    file_names = [file_name for file_name in os.listdir(book_pair_dir)
                  if os.path.isfile(os.path.join(book_pair_dir, file_name))]
    trainer = load_predict_model(model_checkpoint)
    for idx,file_name in enumerate(file_names):
        pair_path = os.path.join(book_pair_dir,file_name)
        anno_pair_path = os.path.join(anno_book_pair_dir,file_name)
        annotate_book(pair_path,anno_pair_path,trainer)
        if idx%100==0:
            logger.info(f'==>>> annotated [{idx}]/[{len(file_names)}] pair files.')


import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dev_set',type=str)
    parser.add_argument('model_checkpoint',type=str)
    parser.add_argument('--process_num',type=int,default=1)
    args = parser.parse_args()

    #book_pair_dir = f'./pg_data_full2/{args.dev_set}/book_pairs_unlabel'
    #anno_book_pair_dir = f'./pg_data_full2/{args.dev_set}/book_pairs_predict'

    ### creating target text for collaborative writing
    book_pair_dir = f'./data/collaborative_writing/{args.dev_set}/book_pairs_unlabel'
    anno_book_pair_dir = f'./data/collaborative_writing/{args.dev_set}/book_pairs_predict'
    if not os.path.exists(anno_book_pair_dir):
        os.makedirs(anno_book_pair_dir)
    if args.process_num == 1:
        annotate_book_corpus(book_pair_dir,anno_book_pair_dir,args.model_checkpoint)
    else:
        ctxt = get_context('spawn')
        annotate_book_corpus_multiprocess(
            book_pair_dir,
            anno_book_pair_dir,
            args.model_checkpoint,
            args.process_num,
            ctxt)