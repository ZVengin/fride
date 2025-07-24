import json, argparse, sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path,'./../dataset')
from annotate_writing_mode import annotate_book_corpus

def anno_format_conversion(sour_file, target_file):
    with open(sour_file) as f:
        data = json.load(f)

    records =[]
    for chapter, paragraphs in data.items():
        for paragraph in paragraphs:
            records.append({
                'target': paragraph['paragraph'],
                'paragraph_index': paragraph['paragraph_index'],
                'idx':len(records)
            })

    with open(target_file, 'w') as f:
        json.dump(records,f,indent=2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sour_dir', type=str, required=True)
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--model_checkpoint', type=str, required=True)
    args = parser.parse_args()

    tmp_dir = os.path.join(args.target_dir, 'tmp')
    for filename in os.listdir(args.sour_dir):
        sour_file = os.path.join(args.sour_dir, filename)
        tmp_file = os.path.join(tmp_dir, filename)
        anno_format_conversion(sour_file, tmp_file)
        annotate_book_corpus(args.tmp_dir, args.target_dir,args.model_checkpoint )
