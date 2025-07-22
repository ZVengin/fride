import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import json,argparse,os

def extract_chapters(epub_path):
    book = epub.read_epub(epub_path)
    book_name = os.path.splitext(os.path.basename(epub_path))[0]
    chapters = []
    para_index=0

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')

            # Try to detect chapter title (h1, h2, h3)
            chapter_title = None
            for tag in ['h1', 'h2', 'h3']:
                header = soup.find(tag)
                if header:
                    chapter_title = header.get_text(strip=True)
                    break

            # Extract all paragraphs
            #paragraphs = [{'paragraph':p.get_text(strip=True)}
            #              for p in soup.find_all('p') if p.get_text(strip=True)]
            paragraphs=[]
            for p in soup.find_all('p'):
                if p.get_text(strip=True):
                    paragraphs.append({'paragraph':p.get_text(strip=True),
                                       'paragraph_index':f'{book_name}-{para_index}',
                                       'book_name':book_name})
                    para_index+=1

            # Only include if there are paragraphs (skip TOC, blank sections, etc.)
            if chapter_title and chapter_title != 'Table of Contents' and paragraphs:
                chapters.append({
                    'chapter': chapter_title or 'Untitled Chapter',
                    'paragraphs': paragraphs
                })

    return chapters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epub_dir', type=str, required=True)
    parser.add_argument('--json_dir', type=str, required=True)
    args = parser.parse_args()

    for book_name in os.listdir(args.epub_dir):
        epub_path = os.path.join(args.epub_dir, book_name)
        chapters_data = extract_chapters(epub_path)
        # Convert to JSON string if needed
        json_output = json.dumps(chapters_data, indent=2, ensure_ascii=False)
        json_path = os.path.join(args.json_dir, book_name.replace('epub','json'))
        # Optionally, write to file
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_output)

    print("Extraction complete. Chapters saved to chapters.json.")
