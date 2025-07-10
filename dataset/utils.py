import json
import spacy
import logging

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

max_word_num = 300
word_num_cluster = [(i, min(i + 20, max_word_num)) for i in range(0, max_word_num, 20)] + [(max_word_num, int(1000000))]


def read_jsonl(file_path):
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            sample = json.loads(line)
            samples.append(sample)
    return samples

def write_jsonl(data,path):
    logger.info('==>>> writing data to jsonl file %s'%path)
    data = '\n'.join([json.dumps(line) for line in data])
    with open(path,'w') as f:
        f.write(data)


def read_text(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(line)
    return data


def assign_label(para_word_num):
    # assign a paragraph with a length label according to the word number
    for i, (l, h) in enumerate(word_num_cluster):
        if l < para_word_num <= h:
            return i


def check_describe_type(root):
    # identify the type of current sentence, 'not-act' or 'act'
    # root is the root node of the sentence, we return the list containing the type of its sub-sentence
    describe_types = []
    describe_type = "action"
    if root.pos_ == "AUX":
        describe_type = "non-action"
    elif root.pos_ == "NOUN":
        describe_type = "non-action"
    elif root.pos_ == "VERB":
        if root.lemma_ == "have":
            for child in root.children:
                if child.dep_ == 'dobj':
                    describe_type = "non-action"
        elif root.lemma_ == "be":
            for child in root.children:
                if child.dep_ in ["attr", "acomp", "prep"]:
                    describe_type = "non-action"
        else:
            for child in root.children:
                if child.dep_ in ["acomp","oprd"]:
                    describe_type = "non-action"
    else:
        describe_type = "action"
    describe_types.append(describe_type)
    for child in root.children:
        if child.dep_ == "conj":
            sub_type = check_describe_type(child)
            describe_types.extend(sub_type)
    return describe_types


def get_root(sent):
    for tok in sent:
        if tok.dep_ == "ROOT":
            root = tok
            break
    return root


def paragraph_type(paragraph,parser,action_threshold,description_threshold):
    # we compute the type of the paragraph based on the type of each sentence
    results = []
    doc = parser(paragraph.replace(';', '.'))
    # print_sent(doc)
    for sent in doc.sents:
        check_result = check_describe_type(get_root(sent))
        results.append(check_result)
    score = sum([sum([item == 'action' for item in group]) / len(group) for group in results]) / len(results)
    if score > action_threshold:
        return 'action'
    elif score < description_threshold:
        return 'non-action'
    else:
        return 'mixed'



def is_utter_quote(text,i):
    if i-1>=0 and i+1<len(text):
        if text[i-1].isalpha() and text[i+1].isalpha():
            return False
    else:
        return True



def has_utterance(text):
    stack = []
    uni_sq_stack,uni_dq_stack,sq_stack,dq_stack=[],[],[],[]
    utters = []
    for i in range(len(text)):
        c = text[i]
        if c == '“':
            uni_dq_stack.append((c, i))
        elif c== "‘":
            uni_sq_stack.append((c, i))
        elif c == '”':
            if len(uni_dq_stack) > 0:
                forward_quote = uni_dq_stack.pop()
                if forward_quote[0] == '“':
                    backward_quote = (c, i)
                    utters.append((forward_quote, backward_quote))
                else:
                    uni_dq_stack.append(forward_quote)
        elif c=='’':
            if len(uni_sq_stack) > 0:
                forward_quote = uni_sq_stack.pop()
                if forward_quote[0] == '‘':
                    backward_quote = (c, i)
                    utters.append((forward_quote, backward_quote))
                else:
                    uni_sq_stack.append(forward_quote)
        elif c=='"':
            if len(dq_stack)>0:
                forward_quote = dq_stack.pop()
                if forward_quote[0] == '"':
                    backward_quote = (c, i)
                    utters.append((forward_quote, backward_quote))
                else:
                    dq_stack.append(forward_quote)
                    dq_stack.append((c,i))
            else:
                dq_stack.append((c,i))

        elif c=="'" and is_utter_quote(text,i):
            if len(sq_stack)>0:
                forward_quote = sq_stack.pop()
                if forward_quote[0] == "'":
                    backward_quote = (c, i)
                    utters.append((forward_quote, backward_quote))
                else:
                    sq_stack.append(forward_quote)
                    sq_stack.append((c,i))
            else:
                sq_stack.append((c,i))

        else:
            continue
    return utters

