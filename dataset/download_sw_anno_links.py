#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import re

try:
    from cookielib import CookieJar
    cj = CookieJar()
    import urllib2
    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
    import urllib
    urlretrieve = urllib.urlretrieve
except ImportError:
    import http.cookiejar
    cj = http.cookiejar.CookieJar()
    import urllib
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj))
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urlretrieve = urllib.request.urlretrieve

from bs4 import BeautifulSoup

import os
import datetime
import json

# If you wanna use some info, write them.
REQUIRED = [
    #    'page',
    #    'epub',
    #    'txt',
    #    'title',
    #    'author',
    #    'genres',
    #    'publish',
    #    'num_words',
    'b_idx',
]
book_dir ="smashword_data/anno"
book_num = 100
search_url_pt = 'https://www.smashwords.com/books/category/1/downloads/0/free/medium/{}'
search_urls = [search_url_pt.format(i) for i in range(0,book_num*2,20)]

num_words_pt = re.compile(r'Words: (\d+)')
pub_date_pt = re.compile(r'Published: ([\w\.]+\s[\d]+,\s[\d]+)')


def main():
    start_time = time.time()
    dataset = []
    sys.stderr.write(str(datetime.datetime.now()) + '\n')

    book_index = 0

    for i, s_url in enumerate(search_urls):
        # print('#', i, s_url)
        response = opener.open(s_url)
        body = response.read()
        soup = BeautifulSoup(body, 'lxml')

        book_links = soup.find_all(class_="library-title")

        for b_link in book_links:
            book_index += 1
            b_url = b_link.get('href')
            response = opener.open(b_url)
            body = response.read()
            soup = BeautifulSoup(body, 'lxml')

            # get meta
            meta_infos = soup.find_all(class_="col-md-3")
            if not meta_infos:
                sys.stderr.write('Failed: meta_info {}\n'.format(b_url))
                continue
            meta_txts = [
                m.text for m in meta_infos
                if 'Language: English' in m.text]

            # check lang
            is_english = len(meta_txts) >= 1
            if not is_english:
                continue

            # get num words
            meta_txt = meta_txts[0].replace(',', '')
            match = num_words_pt.search(meta_txt)
            if match:
                num_words = int(match.group(1))
            elif 'num_words' in REQUIRED:
                sys.stderr.write('Failed: num_words {}\n'.format(b_url))
                continue
            else:
                num_words = 0

            # get publish date
            meta_txt = meta_txts[0]
            match = pub_date_pt.search(meta_txt)
            if match:
                pub_date = match.group(1)
            elif 'publish' in REQUIRED:
                sys.stderr.write('Failed: publish {}\n'.format(b_url))
                continue
            else:
                pub_data = ''

            # get genres
            genre_txts = soup.find_all(class_="category")
            if genre_txts:
                genres = [g.text.replace('\u00a0\u00bb\u00a0', '\t')
                          for g in genre_txts]
            elif 'genres' in REQUIRED:
                sys.stderr.write('Failed: genre {}\n'.format(b_url))
                continue
            else:
                genres = []

            is_fiction=True
            for genre in genres:
                if 'Fiction' not in genre:
                    is_fiction=False
            if not is_fiction or len(genres) == 0:
                continue


            # get title
            title = soup.find("h1")
            if title:
                title = title.text
            elif 'title' in REQUIRED:
                sys.stderr.write('Failed: title {}\n'.format(b_url))
                continue
            else:
                title = ''

            # get author
            author = soup.find(itemprop="author")
            if author:
                author = author.text
            elif 'author' in REQUIRED:
                sys.stderr.write('Failed: author {}\n'.format(b_url))
                continue
            else:
                author = ''

            # get epub
            download_section = soup.find('div',{'id':'download'})
            epub_url = ""
            txt_url = ""
            if download_section is not None:
                tag_as = download_section.find_all('a')
                for tag_a in tag_as:
                    if tag_a.has_attr('href') and tag_a['href'].endswith('txt'):
                       txt_url = 'https://www.smashwords.com'+tag_a['href']
                     
            if not txt_url:
                sys.stderr.write('Failed: epub and txt {}\n'.format(b_url))
                continue

            data = {
                'page': b_url,
                'epub': epub_url,
                'txt': txt_url,
                'title': title,
                'author': author,
                'genres': genres,
                'publish': pub_date,
                'num_words': num_words,
                'b_idx': book_index
            }
            print(json.dumps(data))
            dataset.append(data)
    return dataset



import os
if __name__ == '__main__':
    book_infos = []
    if not os.path.exists(book_dir):
        os.makedirs(book_dir)
    if os.path.exists(os.path.join(book_dir, 'url_list.jsonl')):
        with open(os.path.join(book_dir, 'url_list.jsonl'), "r") as f:
            book_infos = []
            for line in f:
                line = line.strip()
                if line:
                    book_infos.append(json.loads(line))
    max_retries = 100
    retry_count = 0
    while True:
        extract_book_infos = main()
        book_infos += extract_book_infos
        if len(book_infos)> book_num or retry_count>max_retries:
            with open(os.path.join(book_dir, 'url_list.jsonl'), 'w') as f:
                book_infos_str = [json.dumps(book_info) for book_info in book_infos]
                f.write('\n'.join(book_infos_str))
            break
        print(f'collected {len(book_infos)} books!')
        retry_count +=1
