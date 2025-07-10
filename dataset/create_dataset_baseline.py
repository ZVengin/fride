import json
import os
import random
from utils import read_jsonl, assign_label


def select_pairs(book_dir, dataset_dir, pair_nums, dev_set):
    file_names = [file_name for file_name in os.listdir(book_dir)
                  if os.path.isfile(os.path.join(book_dir, file_name))]

    def finish_select():
        is_finish = True
        for k in pair_nums.keys():
            # check the number of samples for each length is larger than the specified one
            if len(dataset[k])<pair_nums[k]:
                is_finish = False
                break
        return is_finish

    # read all book names from the book_pair directory
    # assign idx to each book name and build a set of book idx
    book_idxs = set(range(len(file_names)))
    dataset = {k: [] for k in pair_nums.keys()}
    #each_mode_num = {k: pair_nums[k] // len(dataset[k]) for k in pair_nums.keys()}
    # we iteratively add samples to dataset until the sample number meets requirements, at each turn,
    # we will only add samples from 1000 books
    while (not finish_select()) and len(book_idxs) > 0:
        sel_book_idxs = random.sample(book_idxs, min(1000, len(book_idxs)))
        for book_idx in sel_book_idxs:
            file_path = os.path.join(book_dir, file_names[book_idx])
            samples = read_jsonl(file_path)
            for sample in samples:
                targ_word_num = sample['target_word_num']
                len_label = str(assign_label(targ_word_num))
                if len_label in dataset:
                    dataset[len_label].append(sample)
        book_idxs = book_idxs - set(sel_book_idxs)

    sampled_dataset = []
    for k in dataset.keys():
        sampled_dataset.extend(random.sample(dataset[k],k=pair_nums[k]))
        print(f'take  [{pair_nums[k]}] samples from [{len(dataset[k])}] samples for length [{k}]')

    sampled_dataset = [json.dumps(sample) for sample in sampled_dataset]
    random.shuffle(sampled_dataset)
    dev_set_path = os.path.join(dataset_dir, f'{dev_set}_baseline_{len(sampled_dataset)}.jsonl')
    with open(dev_set_path, 'w') as f:
        f.write('\n'.join(sampled_dataset))
    print(f'saving the dataset to file: {dev_set_path}')



def main(book_dir, result_dir, dev_set):
    #group_sizes = {'train': 30000, 'valid': 120, 'test': 120}
    group_sizes = {'train': 40000, 'valid': 160, 'test': 160}
    pair_nums = {str(i): group_sizes[dev_set] for i in range(1, 10)}
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    select_pairs(book_dir, result_dir, pair_nums, dev_set)


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dev_name', type=str)
    parser.add_argument('cfg_name', type=str)
    args = parser.parse_args()
    dev_set = args.dev_name
    book_dir = f'./pg_data/{dev_set}/book_pairs_{args.cfg_name}'
    result_dir = f'./pg_data/{dev_set}/book_selected_pairs_{args.cfg_name}'
    main(book_dir, result_dir, args.dev_name)
