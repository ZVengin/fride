import json
from model import Model
from dataloader import get_dataloader
from config import *
from utils import logger
from predict import predict_text
import optuna
import random

cache_dir = os.environ['CACHE_DIR'] if os.environ['CACHE_DIR'] else "/work/gk77/k77006/.cache"


def objective(trail):
    model = Model(cfg.model_name, cfg.model_path, cache_dir=cache_dir)
    dataset_config = trail.suggest_categorical("dataset_config",["cfg4","cfg3"])
    train_file = cfg.train_file.replace("config_name",dataset_config)
    validate_file = cfg.validate_file.replace("config_name",dataset_config)
    train_loader = get_dataloader(train_file,
                                  cfg.batch_size,
                                  2)
    train_loader.dataset.data_dicts = random.sample(train_loader.dataset.data_dicts,8100)
    eval_loader = get_dataloader(validate_file,
                                 1, 2)
    eval_loader.dataset.data_dicts = random.sample(eval_loader.dataset.data_dicts, 108)

    model.load_data("train", train_loader)
    model.load_data("test", eval_loader)
    model.add_special_tokens()

    epoch_num = 2 #trail.suggest_int("epoch_num",2,5)
    warmup_proportion = trail.suggest_float("warmup",0.1,0.5)
    train_steps = len(train_loader) * epoch_num
    warmup_steps = int(train_steps * warmup_proportion)
    logger.info(f'train steps in each epoch:{len(train_loader)}, total train steps:{train_steps}, '
                + f'warmup steps:{warmup_steps}')

    lr = trail.suggest_float("lr",1e-5,1e-4)
    model.get_optimizer(lr,
                        train_steps,
                        warmup_steps,
                        cfg.weight_decay,
                        cfg.epsilon)

    model.create_training_log(len(model._dataset["train"])*(epoch_num+1), cfg.log_dir)

    cfg.train_file = train_file
    cfg.validate_file = validate_file
    cfg.epoch = epoch_num
    cfg.warmup_proportion = warmup_proportion
    cfg.lr = lr
    logger.info(f"saving training arguments:{vars(cfg)}...")
    #with open(os.path.join(cfg.log_dir, "training_arguments.json"), "w") as f:
    #    json.dump(vars(cfg), f)

    for epoch in range(epoch_num):
        logger.info(f"==>>> starting epoch [{epoch}]/[{epoch_num}]...")
        model.train_epoch(no_tqdm=True,
                          train_device_map=cfg.train_device_map,
                          eval_device_map=cfg.eval_device_map)
    model._model.split_to_gpus(cfg.eval_device_map)
    predict_text_path = f"{cfg.log_dir}/hyper_predict_text_trail{trail.number}"
    eval_result = predict_text(model,predict_text_path)
    f1_score = eval_result.at['overall_mode_score','F1']
    return f1_score

import argparse

project_dir = os.environ["PROJECT_DIR"] if os.environ[
    "PROJECT_DIR"] else "/work/gk77/k77006/repos/summary2story_booksum"
data_dir = project_dir + "/dataset/pg_data_full2/{}/book_train_{}"

model_paths = {"bart":"facebook/bart-large",
               "gpt2":"gpt2-large",
               "t5":"t5-large"}

model_train_devices = {
    "bart":{0: "model"} if torch.cuda.device_count()<2 else {0:"encoder",1:"decoder"},
    "gpt2":{0: list(range(36))} if torch.cuda.device_count()<2 else {0:list(range(18)),1:list(range(18,36))},
    "t5":{0: list(range(24))} if torch.cuda.device_count()<2 else {0:list(range(12)),1:list(range(12,24))}
}

model_eval_devices = {
    "bart":{0: "model"},
    "gpt2":{0: list(range(36))},
    "t5":{0: list(range(24))}
}

sample_nums ={
    "train":270000,
    "valid":1080,
    "test":1080
}

cfg = TrainingArguments(
        train_file=None,
        validate_file=None,
        test_file=None,
        model_name=None,
        model_path=None,
        batch_size=16,
        epoch=None,
        lr=None,
        warmup_proportion=None,
        weight_decay=0.,
        epsilon=1e-8,
        train_device_map=None,
        eval_device_map=None,
        log_dir=None
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_type", type=str, help="e.g. sentence or keywords")
    parser.add_argument("model_name", type=str, help="e.g. bart or gpt2 or t5")
    parser.add_argument("exp_name",type=str, help="e.g. exp1")
    args = parser.parse_args()
    cfg.train_file = os.path.join(data_dir.format('train','config_name'), f"train_{args.summary_type}_{sample_nums['train']}.jsonl")
    cfg.validate_file = os.path.join(data_dir.format('valid','config_name'), f"valid_{args.summary_type}_{sample_nums['valid']}.jsonl")
    cfg.model_name = args.model_name
    cfg.model_path = model_paths[args.model_name]
    cfg.train_device_map = model_train_devices[args.model_name]
    cfg.eval_device_map = model_eval_devices[args.model_name]
    cfg.log_dir = f"./exp_hyper_search/{args.model_name}_{args.summary_type}/{args.exp_name}"

    study = optuna.create_study(directions=['maximize'])
    study.optimize(objective,n_trials=20)

    trials = sorted(study.best_trials, key=lambda t: t.values)

    for trial in trials:
        print("  Trial#{}".format(trial.number))
        print("    Values: F1 score={}".format(trial.values[0]))
        print("    Params: {}".format(trial.params))
