import json
import os
import spacy
import logging
from utils import paragraph_type, has_utterance
from torch.multiprocessing import Pool

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

link_verbs = ["act", "be", "appear", "become", "come", "do", "fall", "feel", "get", "gp", "grow", "have", "indicate",
              "seem", "keep",
              "look", "prove", "remain", "smell", "sound", "stay", "taste", "turn"]

spacy.prefer_gpu()
parser = spacy.load('en_core_web_lg')
concat_paras = True


def read_file(file_path):
    chapter = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            chapter.append(line)
    return chapter


def label_utterance(chapter):
    para_labels = []
    for para in chapter:
        para_utters = has_utterance(para)

        def update_quote(text, qoute_pairs):
            for f_q, b_q in qoute_pairs:
                if f_q[0] == '"':
                    text = text[:f_q[1]] + '“' + text[f_q[1] + 1:]
                if b_q[0] == '"':
                    text = text[:b_q[1]] + '”' + text[b_q[1] + 1:]
            return text

        para = update_quote(para, para_utters)
        word_num = len(para.split())
        para_labels.append({
            'paragraph': para,
            'label': 'utterance' if len(para_utters) > 0 else 'text',
            'position': para_utters,
            'word_num': word_num
        })
    return para_labels


def create_pair_chapter(file_path):
    chapter = read_file(file_path)
    para_labels = label_utterance(chapter)
    chapter_pairs = []

    def get_context(para_index):
        word_count = 0
        context = ''
        while (para_index > 0 and word_count < 100):
            context = para_labels[para_index]['paragraph'] + context
            word_count += para_labels[para_index]['word_num']
            para_index -= 1
        context = ' '.join(context.split()[-100:])
        return context

    def concatenate_paragraphs(para_labels):
        concat_pairs = []
        for j in range(len(para_labels)):
            para_label = para_labels[j]
            if para_label['label'] == 'text' and 20 <= para_label['word_num'] <= 200:
                k = j + 1
                temp_para = para_label["paragraph"]
                while k < len(para_labels):
                    if para_labels[k]['label'] == 'utterance' or para_labels[k]['word_num']<20:
                        break
                    temp_para = " <newline> ".join([temp_para, para_labels[k]['paragraph']])
                    if 100 <= len(temp_para.split()) <= 200:
                        context = get_context(j - 1)
                        concat_pairs.append({
                            'target': temp_para,
                            'context': context,
                            'target_word_num': len(temp_para.split()),
                            'target_label': 'text',
                            'concat': 'concat'
                        })
                    elif len(temp_para.split()) > 200:
                        break
                    k += 1
        return concat_pairs

    for i in range(len(para_labels)):
        para_label = para_labels[i]
        if para_label['label'] == 'text' and 20 <= para_label['word_num'] <= 200:
            context = get_context(i - 1)
            chapter_pairs.append({
                'target': para_label['paragraph'],
                'context': context,
                'target_word_num': para_label['word_num'],
                'target_label': para_label['label'],
                'concat': 'non-concat'
            })

    if concat_paras:
        chapter_pairs += concatenate_paragraphs(para_labels)

    utter_group = []
    word_count = 0
    for i in range(len(para_labels)):
        para_label = para_labels[i]
        if para_label['label'] == 'utterance' and word_count + para_label['word_num'] <= 200:
            utter_group.append(i)
            word_count += para_label['word_num']
        elif para_label['label'] == 'utterance' and word_count + para_label['word_num'] > 200:
            if len(utter_group) > 0 and word_count >= 20:
                context = get_context(utter_group[0] - 1)
                target_text = ' <newline> '.join([para_labels[k]['paragraph'] for k in utter_group])
                chapter_pairs.append({
                    'target': target_text,
                    'context': context,
                    'target_word_num': word_count,
                    'target_label': 'dialogue',
                    'concat': 'non-concat'
                })
            if para_label['word_num'] <= 200:
                utter_group = [i]
                word_count = para_label['word_num']
            else:
                utter_group = []
                word_count = 0
        else:
            if len(utter_group) > 0:
                if para_label['word_num'] < 20 and word_count + para_label['word_num'] <= 200 and \
                        para_labels[utter_group[-1]]['label'] == 'utterance':
                    utter_group.append(i)
                    word_count += para_label['word_num']
                else:
                    if word_count >= 20:
                        context = get_context(utter_group[0] - 1)
                        target_text = ' <newline> '.join([para_labels[k]['paragraph'] for k in utter_group])
                        chapter_pairs.append({
                            'target': target_text,
                            'context': context,
                            'target_word_num': word_count,
                            'target_label': 'dialogue',
                            'concat': 'non-concat'
                        })
                    utter_group = []
                    word_count = 0
    return chapter_pairs


def label_act(book_pairs, parser, action_threshold, description_threshold):
    """
    :param book_pairs: a group of context-target pairs in the form of [{'context':context,'target':target,'target_label':label},...]
    :param parser: the spacy parser
    :param action_threshold: the minimum proportion of sentences describing actions in the target text
    :param description_threshold: the maximum proportion of sentences describing actions in the target text
    :return: a group of context-target pairs with its label being replaced with dialogue, action or non-action.
    """
    labeled_pairs = []
    logger.info(f'process id:{os.getpid()}, parent id:{os.getppid()}')
    for pair in book_pairs:
        if pair['target_label'] == 'text':
            act_label = paragraph_type(pair['target'], parser, action_threshold, description_threshold)
            pair['target_label'] = act_label
            #if act_label != 'mixed':
            labeled_pairs.append(pair)
        else:
            labeled_pairs.append(pair)
    return labeled_pairs


def create_pair_books(book_dir, pair_dir,config):
    if not os.path.exists(pair_dir):
        os.makedirs(pair_dir)
    for book_idx, book_name in enumerate(os.listdir(book_dir)):
        book_path = os.path.join(book_dir, book_name)
        if os.path.isfile(book_path):
            continue
        book_pairs = []
        logger.info(f'creating book pairs from book:{book_path}')
        for file_name in os.listdir(book_path):
            file_path = os.path.join(book_path, file_name)
            if not os.path.isfile(file_path):
                continue
            chapter_pairs = create_pair_chapter(file_path)
            book_pairs.extend(chapter_pairs)
        # book_pairs = get_label_for_act(book_pairs,parser)
        if config is not None:
            book_pairs = label_act(book_pairs, parser, config['action_threshold'], config['description_threshold'])
        book_pair_path = os.path.join(pair_dir, book_name + '_pair.jsonl')
        with open(book_pair_path, 'w') as f:
            book_pair_strs = [json.dumps(book_pair) for book_pair in book_pairs]
            f.write('\n'.join(book_pair_strs))
        logger.info(f'saving the pairs of book: {book_name} to file :{book_pair_path}')


import argparse

cfgs = {
    'cfg1': {
        'action_threshold': 0.6,
        'description_threshold': 0.35,
        'concat': True
    },
    'cfg2': {
        'action_threshold': 0.7,
        'description_threshold': 0.35,
        'concat': True
    },
    'cfg3': {
        'action_threshold': 0.8,
        'description_threshold': 0.35,
        'concat': True
    },
    'cfg4': {
        'action_threshold': 0.9,
        'description_threshold': 0.35,
        'concat': True
    }

}

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('data_dir',type=str)
    arg_parser.add_argument('dev_name', type=str)
    arg_parser.add_argument('cfg_name', type=str)
    args = arg_parser.parse_args()
    dev_name = args.dev_name
    ### creating the training instances
    book_dir = f'./{args.data_dir}/{dev_name}/book_proc_chapters'
    pair_dir = f'./{args.data_dir}/{dev_name}/book_pairs_{args.cfg_name}'

    ### creating the target text for the collaborative writing
    #book_dir = f'./data/collaborative_writing/{dev_name}/book_proc_chapters'
    #pair_dir = f'./data/collaborative_writing/{dev_name}/book_pairs_{args.cfg_name}'
    cfg = cfgs[args.cfg_name] if args.cfg_name != 'unlabel' else None
    create_pair_books(book_dir, pair_dir, cfg)
