import torch, os, math
import torch.nn as nn
from tqdm import tqdm
from bart_model import BART
from gpt_model import GPT2
from T5_model import T5Model
from utils import logger
from dataset.utils import assign_label
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration

model_mapping = {"gpt2": GPT2, "t5": T5Model, "bart": BART}


class Model(nn.Module):
    def __init__(self, model_name, model_path, cache_dir=None):
        nn.Module.__init__(self)
        self._model = model_mapping[model_name].from_pretrained(model_path,cache_dir=cache_dir)
        self._model.load_tokenizer(model_path,cache_dir=cache_dir)
        self._optimizer = None
        self._global_step = 0
        self._lr_scheduler = None

        self._dataset = {}
        self._eval_steps = None
        self._log_dir = None
        self._log_file = None
        self._best_eval_loss = None

    def set_input(self,add_mode,add_length, add_summary):
        self._model.set_input(add_mode,add_length,add_summary)

    def freeze_parameters(self, freeze=False):
        self._model.freeze_parameters(freeze)

    def split_to_gpus(self, device_map):
        self._model.split_to_gpus(device_map)

    def create_training_log(self, eval_steps, log_dir):
        self._log_dir = log_dir
        self._eval_steps = eval_steps
        self._best_eval_loss = float("inf")

        os.makedirs(os.path.join(self._log_dir, "ckpt_gens"), exist_ok=True)
        self._log_file = open(os.path.join(self._log_dir, "log.txt"), "w")

    def get_optimizer(self, lr, training_steps, warmup_steps,
                      weight_decay, adam_epsilon):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self._model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self._model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0
             }
        ]
        self._optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
        self._lr_scheduler = get_linear_schedule_with_warmup(self._optimizer, num_warmup_steps=warmup_steps,
                                                             num_training_steps=training_steps)

    def save_model(self, model_path):
        self._model.save_model(model_path)
        logger.info(f"==>>> model saved in {model_path}")

    def load_model(self, model_name, model_path, cache_dir=None):
        print(f"==>>> loading model from the file:{model_path}", file=self._log_file)
        self._log_file.flush()
        self._model = model_mapping[model_name].from_pretrained(model_path, cache_dir=cache_dir)
        self._model.load_tokenizer(model_path, cache_dir=cache_dir)

    def load_data(self, split, data_loader):
        self._dataset[split] = data_loader

    def train_epoch(self, no_tqdm=False, train_device_map=None, eval_device_map=None):
        assert "train" in self._dataset
        self._model.split_to_gpus(train_device_map)
        logger.info(f'there are [{len(self._dataset["train"])}] batches...')
        for batch in (self._dataset["train"] if no_tqdm else tqdm(self._dataset["train"])):
            total_batch_num = len(self._dataset["train"])
            self._model.train()
            self._optimizer.zero_grad()
            loss = self._model.get_loss(batch)
            loss.backward()
            batch_loss = loss.item()
            #for sample in batch:
            #    loss = self._model.get_loss([sample])
            #    loss /= len(batch)
            #    loss.backward()
            #    batch_loss += loss.item()
            self._optimizer.step()
            self._lr_scheduler.step()
            self._global_step += 1

            if self._global_step % 10 == 0:
                logger.info(f"==>>>the loss in batch [{self._global_step % total_batch_num}] is: {batch_loss}")

            if self._global_step % self._eval_steps == 0:
                self._model.split_to_gpus(eval_device_map)
                self.generate_log()
                self._model.split_to_gpus(train_device_map)


    def evaluate(self, no_tqdm=False):
        assert "validate" in self._dataset
        self._model.eval()

        loss_list = []
        for batch in (self._dataset["validate"] if no_tqdm else tqdm(self._dataset["validate"])):
            with torch.no_grad():
                loss = self._model.get_loss(batch)
                loss_list.append(loss.item())
        return sum(loss_list) / len(loss_list)

    def generate(self, sample, do_sample=False, num_beams=1, max_length=100):
        self._model.eval()
        inputs = self._model.encode(sample.src_text)
        out_text = self._model.generate(**{"input_ids": inputs["input_ids"].to("cuda"),
                                           "max_length": max_length,
                                           "do_sample": do_sample,
                                           "early_stopping": True,
                                           "num_beams": num_beams,
                                           "eos_token_id": self._model.tokenizer.eos_token_id}
                                        )
        out_text = self._model.tokenizer.decode(out_text.tolist()[0])
        return out_text

    def generate_log(self):
        loss = self.evaluate()
        print(f"==>>> evaluation loss at the global step:[{self._global_step}] is: {loss}",file=self._log_file)

        if loss <= self._best_eval_loss:
            self._best_eval_loss = loss
            self.save_model(os.path.join(self._log_dir, f"best_model.pt"))
            print("==>>>Best model updated.", file=self._log_file)
        self._log_file.flush()

        count = 0
        gen_file = open(os.path.join(self._log_dir, "ckpt_gens", f"step{self._global_step}"), "w")
        for i in range(100):
            sample = self._dataset["validate"].dataset.__getitem__(i)
            length = len(self._model.tokenizer.encode(sample.target, add_special_tokens=False))
            if length>200:
                continue
            gen_text = self._model.server_inference(
                summary=sample.summary,
                context=sample.context,
                length=assign_label(length),
                text_label=sample.target_label
                #do_sample=True,
                #num_beams=1
            )
            logger.info(f"generated_text:{gen_text}")
            gen_text_len = len(self._model.tokenizer.encode(gen_text[0],add_special_tokens=False))
            if abs(length-gen_text_len)<20:
                count +=1
            print(f'{"=" * 40 + "Example:" + str(i) + "=" * 40}\n' +
                  f'context:\n{sample.context}\n' +
                  f'target_length:{length}, target_label:{sample.target_label}, target:\n{sample.target}\n' +
                  f'summary:\n{sample.summary}\n' +
                  f'generation_length:{gen_text_len}, generation:\n{gen_text[0]}\n\n',file=gen_file)
            gen_file.flush()
        print(f"\n\nthere are {count} samples whose generation length is similar to the target length", file=gen_file)
        gen_file.flush()
        gen_file.close()

    def add_special_tokens(self):
        self._model.add_special_tokens()
