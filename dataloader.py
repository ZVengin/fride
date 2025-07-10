import json,re
from torch.utils.data import Dataset,DataLoader
from collections import namedtuple
from utils import assign_label

MAX_CONTEXT_LEN=300
MAX_SUMMARY_LEN=100
MAX_PARAGRAPH_LEN=500

Sample = namedtuple("Sample",["context","summary","target","target_label"])


def load_data(path):
    data_dicts =[]
    with open(path,"r") as f:
        for line in f:
            data_dict = json.loads(line)
            data_dict['target'] = re.sub(r"\[.*?\]", "", data_dict['target'])
            if data_dict['target'].strip():
                data_dicts.append(data_dict)
    #data_dict_idxs = random.sample(range(len(data_dicts)),len(data_dicts))
    #data_dicts = [data_dicts[i] for i in data_dict_idxs]
    return data_dicts



class Summ2Story_Dataset(Dataset):
    def __init__(self,data_path):
        super(Summ2Story_Dataset, self).__init__()
        self.data_dicts = load_data(data_path)

    def __getitem__(self, idx):
        data_dict = self.data_dicts[idx]

        sample = Sample(context=data_dict['context'],
                        summary=data_dict['summary'],
                        target=data_dict['target'],
                        target_label=data_dict['target_label'])
        return sample

    def __len__(self):
        return len(self.data_dicts)



def get_dataloader(data_path,
                   batch_size,
                   num_workers):
    dataset=Summ2Story_Dataset(data_path)#,model_name,tokenizer,mode)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            collate_fn=lambda x:x)
    return dataloader

