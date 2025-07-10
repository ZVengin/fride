import os
import re

def contain_alpha(text):
    has_alpha=False
    for c in text:
        if c.isalpha():
            has_alpha=True
    return has_alpha

def read_file_pg(file_path):
    pat = 'END OF THE PROJECT GUTENBERG EBOOK'
    with open(file_path,'r') as f:
        chapter = []
        para = []
        for line in f:
            line = line.strip()
            match = line.find(pat)
            if match != -1:
                break
            if line:
                if contain_alpha(line):
                    para.append(line)
            else:
                if len(para)>0:
                    chapter.append(para)
                    para=[]
    return chapter

def read_file_smashwords(file_path):
    with open(file_path,'r') as f:
        chapter = []
        for line in f:
            line = line.strip()
            if line and contain_alpha(line):
                chapter.append([line])
    return chapter

def normalize_word(text):
    words = text.split()
    norm_words = []
    p_forward = re.compile(r'[“\(\[] +\w+')
    p_backward = re.compile(r'\w+ +[?\.,!”;\)\]:]')
    for word in words:
        word=word.translate(str.maketrans('_-—','   ')).strip()
        if (p_forward.match(word) is not None) or (p_backward.match(word) is not None):
            word = word.replace(' ','')
        norm_words.append(word)
    return ' '.join(norm_words)



def process_para(para):
    para = ' '.join(para).strip()
    para = normalize_word(para)
    def check_quote(text):
        stack = []
        for i in range(len(text)):
            c = text[i]
            if c=="“":
                stack.append((c,i))
            elif c=="”":
                if len(stack)>0:
                    stack.pop()
            else:
                continue

        for c,i in stack:
            text = text[:i]+" "+text[i+1:]

        return text
    para = check_quote(para)
    return para


def process_chapters(chapter_dir,proc_dir):
    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)
    max_f_n = max([int(f_n.split('.')[0]) for f_n in os.listdir(chapter_dir)])
    for f_n in os.listdir(chapter_dir):
        if int(f_n.split('.')[0]) == max_f_n:
            continue
        f_p = os.path.join(chapter_dir,f_n)
        if os.path.isfile(f_p):
            chapter = read_file_pg(f_p)
            proc_chapter = []
            for para in chapter:
                para = process_para(para)
                proc_chapter.append(para)
            out_p = os.path.join(proc_dir,f_n)
            with open(out_p,'w') as f:
                chapter_text = '\n'.join(proc_chapter)
                f.write(chapter_text)
            print(f'processed file: {f_p}')



def process_books(book_dir,proc_dir):
    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)
    for dir_n in os.listdir(book_dir):
        dir_p = os.path.join(book_dir,dir_n)
        if not os.path.isfile(dir_p) and len(list(os.listdir(dir_p)))>5:
            out_p = os.path.join(proc_dir,dir_n)
            process_chapters(dir_p,out_p)
            print(f'processed book:{dir_n}')

def main(args):
    book_dir = f"./{args.data_dir}/{args.dev_set}/book_chapters"
    proc_dir = f"./{args.data_dir}/{args.dev_set}/book_proc_chapters"
    process_books(book_dir,proc_dir)

import argparse
if __name__=='__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('data_dir',type=str)
    parser.add_argument('dev_set',type=str)
    args=parser.parse_args()
    main(args)