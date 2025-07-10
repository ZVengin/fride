import os,requests,random
import pandas as pd
from bs4 import BeautifulSoup
s = requests.session()
s.keep_alive = False

def get_page(link):
  print(f'retriving webpage from link: {link}')
  page = s.get(link)
  encoding = page.encoding if 'charset' in page.headers.get('content-type','').lower() else None
  soup=BeautifulSoup(page.content,features="html.parser",from_encoding=encoding)
  return soup

def parse_download_page(link):
    page = get_page(link)
    link_table = page.find_all('table','files')[0]
    items = link_table.find_all('tr')
    link = ''
    for item in items:
        td = item.find(lambda tag:tag.name=='td' and tag.a is not None and tag.a.string=='Plain Text UTF-8')
        if td is not None:
            link = "https://www.gutenberg.org"+td.a['href']
            break
    if not link:
        print(f'Failed to retrieve plain text file from link:{link}')

    return link

def filter_book_list(book_list_df):
    link = "https://www.gutenberg.org/ebooks/"
    # filter out invalid books
    book_list_df = book_list_df.loc[book_list_df['Subjects'].notna()]
    book_list_df = book_list_df.loc[(book_list_df['Type']=='Text')
                               & (book_list_df['Language']=='en')
                               & ([bool('fiction' in entr.lower()) for entr in book_list_df['Subjects']])]
    # construct the link to the book info page
    book_links = book_list_df.apply(lambda x:link+str(x['Text#']),axis=1)
    book_list_df['book_link'] = list(book_links)
    download_links = book_list_df.apply(lambda x: parse_download_page(x['book_link']),axis=1)
    book_list_df['download_link']=download_links
    book_list_df = book_list_df.loc[book_list_df['download_link']!='']
    return book_list_df


retrieve_links=True
def main():
    book_list_path = 'pg_data/pg_catalog.csv'
    book_dir = 'pg_data'
    split = {'train':5000,'valid':100,'test':100}
    if retrieve_links:
        book_list_df = pd.read_csv(book_list_path,low_memory=False)
        #done_book_list_df = pd.read_csv(os.path.join(book_dir,'filt_pg_catalog.csv'))
        #done_idxs = list(done_book_list_df['Text#'])
        #book_list_df = book_list_df.loc[~book_list_df['Text#'].isin(done_idxs)]
        #filt_book_list_df = filter_book_list(book_list_df.sample(min(20000-len(done_idxs),len(book_list_df.index))))
        filt_book_list_df = filter_book_list(book_list_df.sample(20000))
        filt_book_list_df.to_csv(os.path.join(book_dir, 'filt_pg_catalog.csv'), index=False)
    else:
        filt_book_list_df = pd.read_csv(os.path.join(book_dir, 'filt_pg_catalog.csv'),low_memory=False)
    if len(filt_book_list_df.index)< sum([v for k,v in split.items()]):
        print('not enough books for splitting')
        return
    idxs = list(filt_book_list_df.index)
    idxs = random.sample(idxs,len(idxs))
    train_df = filt_book_list_df.loc[idxs[:split['train']]]
    valid_df = filt_book_list_df.loc[idxs[split['train']:split['train']+split['valid']]]
    test_df = filt_book_list_df.loc[idxs[split['train']+split['valid']:]]
    dataset = {
        'train':train_df,
        'valid':valid_df,
        'test':test_df
    }
    for k,v in dataset.items():
        if not os.path.exists(os.path.join(book_dir,k)):
            os.makedirs(os.path.join(book_dir,k))
        path = os.path.join(book_dir,k,'book_info_list.csv')
        v.to_csv(path,index=False)
        print(f'saving {k} set to file:{path}')

main()