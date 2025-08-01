import json, argparse, sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path,'./../dataset'))
from annotate_writing_mode import annotate_book_corpus,annotate_book_corpus_multiprocess

def anno_format_conversion(sour_file, target_file):
    with open(sour_file) as f:
        data = json.load(f)

    records =[]
    for chapter in data:
        for paragraph in chapter['paragraphs']:
            records.append({
                'target': paragraph['paragraph'],
                'paragraph_index': paragraph['paragraph_index'],
                'idx':len(records)
            })

    with open(target_file, 'w') as f:
        records = '\n'.join([json.dumps(r) for r in records])
        f.write(records)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sour_dir', type=str, required=True)
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--proc_num', type=int, default=4)
    parser.add_argument('--model_checkpoint', type=str, required=True)
    args = parser.parse_args()

    tmp_dir = os.path.join(args.target_dir, 'tmp')

    os.makedirs(args.target_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    for filename in os.listdir(args.sour_dir):
        sour_file = os.path.join(args.sour_dir, filename)
        tmp_file = os.path.join(tmp_dir, filename)
        anno_format_conversion(sour_file, tmp_file)
    annotate_book_corpus(tmp_dir, args.target_dir,args.model_checkpoint )

    #ctxt = get_context('spawn')
    #annotate_book_corpus_multiprocess(
    #    book_pair_dir,
    #    anno_book_pair_dir,
    #    args.model_checkpoint,
    #    args.process_num,
    #    ctxt)
