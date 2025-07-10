import torch, json, os
from model import Model
from dataloader import get_dataloader
from config import *
from utils import logger

cache_dir = os.environ['CACHE_DIR'] if 'CACHE_DIR' in os.environ else "/work/gk77/k77006/.cache"


def main(config):
    model = Model(config.model_name, config.model_path, cache_dir=cache_dir)
    model.set_input(add_mode=config.add_mode,add_length=config.add_length,add_summary=config.add_summary)
    train_loader = get_dataloader(config.train_file,
                                  config.batch_size,
                                  2)
    eval_loader = get_dataloader(config.validate_file,
                                 1, 2)

    model.load_data("train", train_loader)
    model.load_data("validate", eval_loader)
    model.add_special_tokens()

    train_steps = len(train_loader) * config.epoch
    warmup_steps = int(train_steps * config.warmup_proportion)
    logger.info(f'train steps in each epoch:{len(train_loader)}, total train steps:{train_steps}, warmup steps:{warmup_steps}')
    model.get_optimizer(config.lr,
                        train_steps,
                        warmup_steps,
                        config.weight_decay,
                        config.epsilon)

    model.create_training_log(len(model._dataset["train"])//5, config.log_dir)
    # model.create_training_log(10, config.log_dir)

    logger.info(f"saving training arguments:{vars(config)}...")
    with open(os.path.join(config.log_dir, "training_arguments.json"), "w") as f:
        json.dump(vars(config), f)

    for epoch in range(config.epoch):
        logger.info(f"==>>> starting epoch [{epoch}]/[{config.epoch}]...")
        model.train_epoch(no_tqdm=True,
                          train_device_map=config.train_device_map,
                          eval_device_map=config.eval_device_map)


import argparse

project_dir = os.environ["PROJECT_DIR"] if "PROJECT_DIR" in os.environ else "/work/gk77/k77006/repos/summary2story_booksum"
data_dir = project_dir + "/dataset/pg_data/{}/book_train_{}"

model_paths = {"bart":"facebook/bart-large",
               "gpt2":"gpt2-large",
               "t5":"t5-large"}

model_train_devices = {
    "bart":{"encoder":"cuda:0","decoder":"cuda:1"},
    "gpt2":{0: list(range(36))} if torch.cuda.device_count()<2 else dict([(idx,list(range(l,min(l+5,36)))) for idx,l in enumerate(range(0,36,5))]),
    "t5":{0: list(range(24))} if torch.cuda.device_count()<2 else dict([(idx,list(range(l,min(l+3,24)))) for idx,l in enumerate(range(0,24,3))])
}

model_eval_devices = {
    "bart":{"encoder":"cuda:0","decoder":"cuda:0"},
    "gpt2":{0: list(range(36))} ,#if torch.cuda.device_count()<2 else dict([(idx,list(range(l,l+5))) for idx,l in enumerate(range(0,36,5))]),
    "t5":{0: list(range(24))} #if torch.cuda.device_count()<2 else dict([(idx,list(range(l,l+3))) for idx,l in enumerate(range(0,24,3))])
}

lr_bsz ={
    "bart":{'lr':4e-5,'batch_size':16},
    "gpt2":{'lr':4e-5,'batch_size':32},
    "t5":{'lr':4e-5,'batch_size':64}
}

sample_nums ={
    "train":360000,
    "valid":1440,
    "test":1440
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_type", type=str, help="e.g, sentence or keywords")
    parser.add_argument("model_name", type=str, help="e.g. bart or gpt2 or t5")
    parser.add_argument("config_name",type=str,help="e.g, cfg4")
    parser.add_argument("exp_name",type=str, help="e.g, exp1")
    parser.add_argument("--add_mode", action="store_true")
    parser.add_argument("--add_length",action="store_true")
    parser.add_argument("--add_summary", action="store_true")
    parser.add_argument("--exp_type",default="",help="e.g. baseline or context2sum")

    args = parser.parse_args()
    exp_type_str = f'_{args.exp_type}_' if args.exp_type != "" else '_'
    cfg = TrainingArguments(
        train_file=os.path.join(data_dir.format('train',args.config_name), f"train_{args.summary_type}{exp_type_str}{sample_nums['train']}.jsonl"),
        validate_file=os.path.join(data_dir.format('valid',args.config_name), f"valid_{args.summary_type}{exp_type_str}{sample_nums['valid']}.jsonl"),
        test_file=None,
        model_name=args.model_name,
        model_path=model_paths[args.model_name],
        add_mode=args.add_mode,
        add_length=args.add_length,
        add_summary=args.add_summary,
        batch_size=lr_bsz[args.model_name]['batch_size'],
        epoch=5,
        lr=lr_bsz[args.model_name]['lr'] if "context2sum" not in args.exp_type else 1e-5,
        warmup_proportion=0.1,
        weight_decay=0.,
        epsilon=1e-8,
        train_device_map=model_train_devices[args.model_name],
        eval_device_map=model_eval_devices[args.model_name],
        log_dir=f"./exp/{args.exp_type+'_' if  'context2sum' in args.exp_type else ''}"\
                +f"{args.model_name}_{args.summary_type}_{args.config_name}"\
                +f"{'_nosum' if not args.add_summary else ''}{'_mode' if args.add_mode else ''}"\
                +f"{'_length' if args.add_length else ''}/{args.exp_name}"
    )

    main(cfg)
