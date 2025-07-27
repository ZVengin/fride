import json,os,argparse
from collections import Counter, defaultdict

def construct_instance(data, context_window_size=1):
    instances = []
    for i in range(1, len(data)):
        context = data[max(i-context_window_size,0):i]
        context_text = '\n'.join([p.get('paragraph') for p in context])
        target = data[i]
        target_label = target.get('target_label')
        target_text = target.get('paragraph')
        instances.append({
            'context': context_text,
            'mode':target_label,
            'target':target_text,
            'paragraph_index': target.get('paragraph_index'),
            'idx': target.get('idx')
        })
    return instances



def filter_instances(instances, min_len=50):
    filtered_instances = []
    for instance in instances:
        context_len = len(instance['context'].split())
        target_len = len(instance['target'].split())
        if context_len >= min_len and target_len >= min_len:
            filtered_instances.append(
                instance
            )
    return filtered_instances


def extract_dialogue(paras):
    new_parags =[]
    dialogue=[]
    for para in paras:
        if para.get('target_label') == 'Dialogue':
            dialogue.append(para)
        else:
            if len(dialogue)>0:
                new_para = {
                    'idx':dialogue[0].get('idx'),
                    'paragraph_index':dialogue[0].get('paragraph_index'),
                    'paragraph': '\n'.join([p.get('paragraph') for p in dialogue]),
                    'target_label':'Dialogue'
                }
                new_parags.append(new_para)
                dialogue = []
            new_parags.append(para)
    if len(dialogue) > 0:
        new_para = {
            'idx': dialogue[0].get('idx'),
            'paragraph_index': dialogue[0].get('paragraph_index'),
            'paragraph': '\n'.join([p.get('paragraph') for p in dialogue]),
            'target_label': 'Dialogue'
        }
        new_parags.append(new_para)
    return new_parags




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sour_dir', type=str, required=True)
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--context_window_size', type=int, default=1)
    parser.add_argument('--min_len', type=int, default=100)
    args = parser.parse_args()
    os.makedirs(args.target_dir, exist_ok=True)
    dataset=[]
    for filename in os.listdir(args.sour_dir):
        if os.path.isfile(os.path.join(args.sour_dir, filename)):
            with open(os.path.join(args.sour_dir, filename), 'r') as f:
                data = []
                for line in f.readlines():
                    line = json.loads(line)
                    if line.get('paragraph') is not None:
                        data.append(line)
            print(len(data))
            data = extract_dialogue(data)
            instances = construct_instance(data, context_window_size=args.context_window_size)
            filtered_instances = filter_instances(instances)
            dataset+=filtered_instances

    mode_counter = Counter([inst.get('mode') for inst in dataset])
    min_mode_num = min(mode_counter.values())
    mode_to_insts = defaultdict(list)
    for inst in dataset:
        mode_to_insts[inst['mode']].append(inst)

    new_dataset = []
    for mode in ['Dialogue','Description','Action']:
        new_dataset += random.sample(mode_to_insts[mode],k=min_mode_num)

    with open(os.path.join(args.target_dir, 'eval_dataset.json'), 'w') as f:
        json.dump(new_dataset,f, indent=2)
