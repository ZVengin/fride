import os
import torch


class TrainingArguments:
    def __init__(self,
                 train_file,
                 validate_file,
                 test_file,
                 model_name,
                 model_path,
                 batch_size,
                 epoch,
                 lr,
                 warmup_proportion,
                 weight_decay,
                 epsilon,
                 train_device_map,
                 eval_device_map,
                 log_dir,
                 add_mode=None,
                 add_length=None,
                 add_summary=None):
        self.train_file = train_file
        self.validate_file = validate_file
        self.test_file = test_file
        self.model_name = model_name
        self.model_path = model_path
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.warmup_proportion = warmup_proportion
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.train_device_map = train_device_map
        self.eval_device_map = eval_device_map
        self.log_dir = log_dir
        self.add_mode = add_mode
        self.add_length = add_length
        self.add_summary=add_summary


class TestArguments:
    def __init__(self,
                 test_file,
                 model_name,
                 model_path,
                 batch_size,
                 lr):
        self.train_file = test_file
        self.model_name = model_name
        self.model_path = model_path
        self.batch_size = batch_size
        self.lr = lr


project_dir = os.environ["PROJECT_DIR"] if os.environ[
    "PROJECT_DIR"] else "/work/gk77/k77006/repos/summary2story_booksum"
data_dir = project_dir + "/dataset/pg_data_full2/{}/book_train"

bart_sent_config = TrainingArguments(
    train_file=os.path.join(data_dir.format('train'), "train_sentence_270000.jsonl"),
    validate_file=os.path.join(data_dir.format('valid'), "valid_sentence_2700.jsonl"),
    test_file=os.path.join(data_dir.format('test'), "test_sentence_2700.jsonl"),
    model_name="bart",
    model_path="facebook/bart-large",
    batch_size=16,
    epoch=2,
    lr=4e-5,
    warmup_proportion=0.1,
    weight_decay=0.,
    epsilon=1e-8,
    train_device_map={0: "model"} if torch.cuda.device_count()<2 else {0:"encoder",1:"decoder"},
    eval_device_map={0: "model"},
    log_dir="./exp2/bart_sent/exp1"
)

gpt2_sent_config = TrainingArguments(
    train_file=os.path.join(data_dir.format('train'), "train_sentence_270000.jsonl"),
    validate_file=os.path.join(data_dir.format('valid'), "valid_sentence_2700.jsonl"),
    test_file=os.path.join(data_dir.format('test'), "test_sentence_2700.jsonl"),
    model_name="gpt2",
    model_path="gpt2-large",
    batch_size=16,
    epoch=2,
    lr=4e-5,
    warmup_proportion=0.1,
    weight_decay=0.,
    epsilon=1e-8,
    train_device_map={0: list(range(36))} if torch.cuda.device_count()<2 else {0:list(range(18)),1:list(range(18,36))},  # {0:[0,1,2,3,4,5,6,7,8,9,10,11]},#{0:[0,1,2,3,4,5],1:[6,7,8,9,10,11]},
    eval_device_map={0: list(range(36))},  # {0:[0,1,2,3,4,5,6,7,8,9,10,11]},
    log_dir="./exp2/gpt2_sent/exp1"
)

t5_sent_config = TrainingArguments(
    train_file=os.path.join(data_dir.format("train"), "train_sentence_270000.jsonl"),
    validate_file=os.path.join(data_dir.format("valid"), "valid_sentence_2700.jsonl"),
    test_file=os.path.join(data_dir.format('test'), "test_sentence_2700.jsonl"),
    model_name="t5",
    model_path="t5-large",
    batch_size=16,
    epoch=2,
    lr=4e-5,
    warmup_proportion=0.1,
    weight_decay=0.,
    epsilon=1e-8,
    train_device_map={0: list(range(12)), 1: list(range(12,24))},
    eval_device_map={0: list(range(24))},
    log_dir="./exp2/t5_sent/exp1"
)

bart_kws_config = TrainingArguments(
    train_file=os.path.join(data_dir.format("train"), "train_keywords_270000.jsonl"),
    validate_file=os.path.join(data_dir.format("valid"), "valid_keywords_2700.jsonl"),
    test_file=os.path.join(data_dir.format('test'), "test_keywords_2700.jsonl"),
    model_name="bart",
    model_path="facebook/bart-large",
    batch_size=16,
    epoch=2,
    lr=4e-5,
    warmup_proportion=0.1,
    weight_decay=0.,
    epsilon=1e-8,
    train_device_map={0: "encoder", 1: "decoder"},
    eval_device_map={0: "model"},
    log_dir="./exp2/bart_kws/exp1"
)

gpt2_kws_config = TrainingArguments(
    train_file=os.path.join(data_dir.format("train"), "train_keywords_270000.jsonl"),
    validate_file=os.path.join(data_dir.format("valid"), "valid_keywords_2700.jsonl"),
    test_file=os.path.join(data_dir.format('test'), "test_keywords_2700.jsonl"),
    model_name="gpt2",
    model_path="gpt2-large",
    batch_size=16,
    epoch=2,
    lr=4e-5,
    warmup_proportion=0.1,
    weight_decay=0.,
    epsilon=1e-8,
    train_device_map={0: list(range(36))} if torch.cuda.device_count()<2 else {0:list(range(18)),1:list(range(18,36))},
    eval_device_map={0: list(range(36))},
    log_dir="./exp2/gpt2_kws/exp1"
)

t5_kws_config = TrainingArguments(
    train_file=os.path.join(data_dir.format("train"), "train_keywords_270000.jsonl"),
    validate_file=os.path.join(data_dir.format("valid"), "valid_keywords_2700.jsonl"),
    test_file=os.path.join(data_dir.format('test'), "test_keywords_2700.jsonl"),
    model_name="t5",
    model_path="t5-large",
    batch_size=16,
    epoch=2,
    lr=4e-5,
    warmup_proportion=0.1,
    weight_decay=0.,
    epsilon=1e-8,
    train_device_map={0: list(range(12)), 1: list(range(12,24))},
    eval_device_map={0: list(range(24))},
    log_dir="./exp2/t5_kws/exp1"
)
