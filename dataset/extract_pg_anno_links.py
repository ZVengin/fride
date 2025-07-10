import pandas as pd
import random,os
from extract_pg_links import filter_book_list

book_infos = pd.read_csv('pg_data/pg_catalog.csv')
book_idxs = set(book_infos['Text#'].tolist())

selected_idxs = []
path = 'pg_data/{}/book_info_list.csv'
for dev in ['train','valid','test']:
    if os.path.exists(path.format(dev)):
        data = pd.read_csv(path.format(dev))
        selected_idxs += data['Text#'].tolist()
unselected_idxs = book_idxs - set(selected_idxs)

unselected_idxs = random.sample(unselected_idxs,3000)
unselected_book_infos = list(map(lambda x: x in unselected_idxs,
                                 book_infos['Text#'].tolist()))
filt_book_info_df = filter_book_list(book_infos[unselected_book_infos])
#anno_dir='pg_data_full2/anno2' #the supplement anno set
anno_dir='pg_data/anno'
if not os.path.exists(anno_dir):
    os.makedirs(anno_dir)
filt_book_info_df.to_csv(os.path.join(anno_dir,'book_info_list.csv'),index=False)

