import argparse
import re
import time
import os
import pandas as pd
from progressbar import ProgressBar
from glob import glob
import sys
import json
import wget
import string

parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', '--out', type=str, required=True)
parser.add_argument('--list-path', '--list', type=str, required=True)
args = parser.parse_args(['--list','pg_data/anno/book_info_list.csv','--out','pg_data/anno/book_txts'])


def main(args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    book_info_list = pd.read_csv(args.list_path)
    done_files = set([os.path.split(path)[-1]
                      for path in glob(os.path.join(args.out_dir, '*.txt'))])
    for idx in book_info_list.index:
        book_title = book_info_list.at[idx,'Title']
        book_title = re.sub(r"[^a-zA-Z0-9 ]","",book_title)
        #book_title = book_title.translate(str.maketrans(string.punctuation," "*len(string.punctuation)))
        book_title = '-'.join(book_title.split())
        book_id = book_info_list.at[idx,'Text#']

        out_file_name = '{}__{}.txt'.format(
            book_id, book_title)
        out_path = os.path.join(args.out_dir, out_file_name)
        if out_file_name in done_files:
            continue
        download_link = book_info_list.at[idx,'download_link']
        try:
            wget.download(download_link, out_path)  # download epub
            print(f'downloaded book: {book_title}')

        except Exception as e:
            sys.stderr.write(str(e) + '\n')
            if os.path.exists(out_path):
                os.remove(out_path)



"""
def main():
    dataset = []
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filelist_path = args.list_path

    lines = list(open(filelist_path).readlines())

    done_files = set([os.path.split(path)[-1]
                      for path in glob(os.path.join(out_dir, '*.txt'))])
    sys.stderr.write('{} files already had been saved in {}.\n'.format(
        len(done_files), out_dir))

    for i, line in enumerate(ProgressBar()(lines)):
        if not line.strip():
            continue
        # read data
        try:
            # {"page": "https://www.smashwords.com/books/view/52", "epub": "https://www.smashwords.com/books/download/52/8/latest/0/0/smashwords-style-guide.epub", "title": "Smashwords Style Guide", "author": "Mark Coker", "genres": ["Nonfiction\tComputers and Internet\tDigital publishing", "Nonfiction\tPublishing\tSelf-publishing"], "publish": "May 05, 2008", "num_words": 28300, "b_idx": 1}
            data = json.loads(line.strip())
            book_id = data['EBook-No.'][0]
            file_link = data['download_link']
            file_name = data['Title'][0].translate(str.maketrans(string.punctuation," "*len(string.punctuation)))
            file_name = '-'.join(file_name.split())

            out_file_name = '{}__{}.txt'.format(
                book_id, file_name)
            out_path = os.path.join(out_dir, out_file_name)
            if out_file_name in done_files:
                continue
            wget.download(file_link, out_path)  # download epub
            print(f'downloaded book: {file_name}')

        except Exception as e:
            sys.stderr.write(str(e) + '\n')
            if os.path.exists(out_path):
                os.remove(out_path)
"""

main(args)