import json
import os
import random
from utils import read_jsonl, assign_label


def select_pairs(book_dir, dataset_dir, pair_nums, dev_set, modes):
    file_names = [file_name for file_name in os.listdir(book_dir)
                  if os.path.isfile(os.path.join(book_dir, file_name))]

    def finish_select():
        is_finish = True
        for k in pair_nums.keys():
            # get the minimum group size in all types of description groups
            min_group_size = min([len(dataset[k][label]) for label in dataset[k].keys()])
            if min_group_size * len(dataset[k]) < pair_nums[k]:
                is_finish = False
                break
        return is_finish

    # read all book names from the book_pair directory
    # assign idx to each book name and build a set of book idx
    book_idxs = set(range(len(file_names)))
    #
    dataset = {k: {l: [] for l in modes} for k in pair_nums.keys()}
    each_mode_num = {k: pair_nums[k] // len(dataset[k]) for k in pair_nums.keys()}
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
                mode_label = sample['target_label']
                if len_label in dataset and mode_label in modes:
                    dataset[len_label][mode_label].append(sample)
        book_idxs = book_idxs - set(sel_book_idxs)

    for k in dataset.keys():
        for l in dataset[k].keys():
            print(f'length:{k}, mode:{l}, num:{len(dataset[k][l])}')

    # sel_pair_nums = {k:{l:len(dataset[k][l]) for l in dataset[k].keys()} for k in dataset.keys()}

    # if not finish_select():
    #    print(f'the data pairs are not enough to create the dataset, already selected pairs: {sel_pair_nums}')
    #    dataset = sum(
    #        [sum(
    #            [random.sample([s for s in dataset[k][l] if s['concat']==],each_mode_num[k]) if len(dataset[k][l])>each_mode_num[k]
    #             else random.choices(dataset[k][l],k=each_mode_num[k]) for l in dataset[k].keys()],[]
    #        ) for k in dataset.keys()],[]
    #    )
    # else:
    #    print(f'we totally find {sel_pair_nums} samples')
    #    dataset = sum(
    #        [sum([random.sample(dataset[k][l], each_mode_num[k]) for l in dataset[k].keys()], [])
    #         for k in dataset.keys()], [])

    sampled_dataset = []
    sampled_statistic = {}
    for k in dataset.keys():
        sampled_statistic[k] = {}
        for l in dataset[k].keys():
            concat_samples, non_concat_samples = [], []
            sampled_statistic[k][l] = {}
            for s in dataset[k][l]:
                if s['concat'] == 'concat':
                    concat_samples.append(json.dumps(s))
                else:
                    non_concat_samples.append(json.dumps(s))
            #print(f'total:{len(dataset[k][l])}, non-concat:{len(non_concat_samples)}')
            sampled_statistic[k][l]['total'] = len(dataset[k][l])
            sample_size = min(each_mode_num[k], len(non_concat_samples))
            sampled_statistic[k][l]['non-concat'] = sample_size
            samples = random.sample(non_concat_samples, k=sample_size)
            sample_size = min(len(concat_samples), max(each_mode_num[k] - len(samples), 0))
            sampled_statistic[k][l]['concat'] = sample_size
            samples += random.sample(concat_samples, k=sample_size)
            sample_size = max(0, each_mode_num[k] - len(samples))
            sampled_statistic[k][l]['resample'] = sample_size
            print(f'resample non-concat:{len(non_concat_samples)}')
            samples += random.choices(non_concat_samples+concat_samples, k=sample_size)
            sampled_dataset += samples

    if not finish_select():
        print(f'the data pairs are not enough to create the dataset, already selected pairs: {sampled_statistic}')
    else:
        print(f'we totally find {sampled_statistic} samples')

    random.shuffle(sampled_dataset)
    #sampled_dataset = [json.dumps(s) for s in sampled_dataset]
    dev_set_path = os.path.join(dataset_dir, f'{dev_set}_{len(sampled_dataset)}.jsonl')
    with open(dev_set_path, 'w') as f:
        f.write('\n'.join(sampled_dataset))
    print(f'saving the dataset to file: {dev_set_path}')



def main(book_dir, result_dir, dev_set, modes):
    group_sizes = {'train': 10000*len(modes), 'valid': 40*len(modes), 'test': 40*len(modes),'anno2':100*len(modes)}
    pair_nums = {str(i): group_sizes[dev_set] for i in range(1, 10)}
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    select_pairs(book_dir, result_dir, pair_nums, dev_set, modes)


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dev_name', type=str)
    parser.add_argument('cfg_name', type=str)
    parser.add_argument('writing_mode',type=str)
    args = parser.parse_args()
    dev_set = args.dev_name
    modes = [m.strip() for m in args.writing_mode.split(',')]
    book_dir = f'./pg_data/{dev_set}/book_pairs_{args.cfg_name}'
    result_dir = f'./pg_data/{dev_set}/book_selected_pairs_{args.cfg_name}'
    main(book_dir, result_dir, args.dev_name, modes)
