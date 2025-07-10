import torch
import torch.nn as nn
from utils import logger, max_summary, max_context, max_target,MODES
from transformers import GPT2Tokenizer

GPT_MAX_LEN = 1024

from transformers import GPT2LMHeadModel



class GPT2(nn.Module):
    def __init__(self):
        super(GPT2, self).__init__()
        self.model = None
        self.tokenizer = None
        self.add_length = None
        self.add_mode = None
        self.add_summary=None

    def set_input(self, add_mode, add_length, add_summary):
        self.add_mode = add_mode
        self.add_length = add_length
        self.add_summary = add_summary

    def freeze_parameters(self,freeze=False):
        for p in self.model.transformer.wte.parameters():
            p.requires_grad = not freeze
        for p in self.model.transformer.wpe.parameters():
            p.requires_grad = not freeze

        for m in list(self.model.transformer.h)[:23]:
            for p in m.parameters():
                p.requires_grad = not freeze

    def split_to_gpus(self, mapping=None):
        if mapping is None:
            return
        self.model.parallelize(mapping)

    def load_tokenizer(self, path,cache_dir=None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(path,cache_dir=cache_dir)

    def encode(self, text):
        # GPT2 tokenizer will not add special tokens
        # to the start and the end of text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=GPT_MAX_LEN,
            return_tensors="pt")
        return inputs

    def decode(self, token_ids):
        text = self.tokenizer.decode(token_ids)
        return text

    def prepare_inputs(self, sample):
        context = self.tokenizer.decode(self.tokenizer.encode(
            sample.context,
            add_special_tokens=False
        )[-max_context:])
        #context_token_ids = self.tokenizer.encode(context, add_special_tokens=False)
        if self.add_summary:
            summary = self.tokenizer.decode(self.tokenizer.encode(
                sample.summary,
                add_special_tokens=False
            )[:max_summary])
        target = self.tokenizer.decode(self.tokenizer.encode(
            sample.target,
            add_special_tokens=False
        )[:max_target])
        target_label = sample.target_label

        src_text = (f"{self.tokenizer.additional_special_tokens[3]*4} {self.tokenizer.additional_special_tokens[0]} "
                      if self.add_length else "")\
                   + f"{context} "\
                   + (f"{self.tokenizer.additional_special_tokens[1]} {summary} "
                      if self.add_summary else "") \
                   + (f"{self.tokenizer.additional_special_tokens[2]} {target_label} "
                      if self.add_mode else "")\
                   + f"{self.tokenizer.sep_token} {target}{self.tokenizer.eos_token}"
        src_length_cluster = None
        if self.add_length:
            src_length = len(self.tokenizer.tokenize(src_text))
            src_length_cluster = src_length//20
            src_length_str = self.tokenizer.decode(
                self.tokenizer.encode(str(src_length_cluster)+self.tokenizer.additional_special_tokens[3]*4,
                                      add_special_tokens=False)[:4])
            src_text = src_text.replace(self.tokenizer.additional_special_tokens[3]*4,src_length_str)
            src_text = src_text + ((src_length_cluster+1)*20-src_length)*self.tokenizer.eos_token


        return src_text, src_length_cluster

    def get_loss(self, batch):
        src_texts, src_lengths = [],[]
        for sample in batch:
            src_text, src_length = self.prepare_inputs(sample)
            src_texts.append(src_text)
            src_lengths.append(src_length)
        logger.info(f'[train] src_text:\n{src_text}')
        #logger.info(f'pad_src_text_length:{len(self.tokenizer.tokenize(src_text))},\n'
        #            f'expect_src_text_length:{(src_length + 1) * 20}')
        encoder_inputs = self.encode(src_texts)
        labels = encoder_inputs["input_ids"]
        encoder_inputs = {
            "input_ids": encoder_inputs["input_ids"].to(self.model.transformer.wte.weight.device),
            "attention_mask": encoder_inputs["attention_mask"].to(self.model.transformer.wte.weight.device),
            "labels": labels.to(self.model.lm_head.weight.device)}
        outputs = self.forward(encoder_inputs)
        return outputs.loss

    def forward(self, inputs):
        return self.model.forward(**inputs)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    @classmethod
    def from_pretrained(cls, path,cache_dir=None):
        gpt2 = GPT2LMHeadModel.from_pretrained(path,cache_dir=cache_dir)
        model = GPT2()
        model.model = gpt2
        #model.model.prepare_inputs_for_generation=prepare_inputs_for_generation
        return model

    def generate(self, **inputs):
        return self.model.generate(**inputs)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()



    def add_special_tokens(self):
        assert self.tokenizer is not None
        self.tokenizer.add_special_tokens({
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "sep_token": "<sep>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "additional_special_tokens": ["<sep_1>", "<sep_2>","<sep_3>","<empty>","<newline>"]
        })
        logger.info(f"{len(self.tokenizer.additional_special_tokens)} tokens is added to the vocabulary...")
        logger.info(f"added additional special tokens:{self.tokenizer.additional_special_tokens}")
        self.model.resize_token_embeddings(len(self.tokenizer))


    def prepare_inference_input(self,
                                summary=None,
                                context=None,
                                length=None,
                                text_label=None):
        context = self.tokenizer.decode(self.tokenizer.encode(
            context,
            add_special_tokens=False
        )[-max_context:])
        if self.add_summary:
            summary = self.tokenizer.decode(self.tokenizer.encode(
                summary,
                add_special_tokens=False
            )[:max_summary])
        target_label = text_label
        if self.add_length:
            target_length = (int(length) + 1) * 20
        src_text = (f"{self.tokenizer.additional_special_tokens[3] * 4} {self.tokenizer.additional_special_tokens[0]} "
                      if self.add_length else "") \
                   + f"{context} " \
                   + (f"{self.tokenizer.additional_special_tokens[1]} {summary} "
                      if self.add_summary else "") \
                   + (f"{self.tokenizer.additional_special_tokens[2]} {target_label} "
                      if self.add_mode else "") \
                   + f"{self.tokenizer.sep_token}"
        if self.add_length:
            src_length_cluster = (len(self.tokenizer.tokenize(src_text))+target_length) // 20
            src_length_str = self.tokenizer.decode(
                self.tokenizer.encode(str(src_length_cluster) + self.tokenizer.additional_special_tokens[3] * 4,
                                      add_special_tokens=False)[:4])
            src_text = src_text.replace(self.tokenizer.additional_special_tokens[3]*4, src_length_str)
        logger.info(f'src text in inference of gpt2:{src_text}')
        return src_text

    def server_inference(self,
                         summary=None,
                         context=None,
                         length=None,
                         text_label=None,
                         max_length=512,
                         do_sample=True,
                         top_k=None,
                         top_p=0.9,
                         no_repeat_ngram_size=0,
                         num_beams=1):

        src_text = self.prepare_inference_input(summary=summary,
                                                context=context,
                                                length=length,
                                                text_label=text_label)

        #logger.info(f'[inference] src_text:\n{src_text}')
        inputs = self.encode(src_text)
        logger.info("==>>> inference in gpt model...")
        out_seqs = self.generate(
            **{"input_ids": inputs["input_ids"].to(self.model.device),
               "max_length": max_length,
               "do_sample": do_sample,
               "early_stopping": True,
               "num_beams": num_beams,
               "top_k": top_k,
               "top_p": top_p,
               # "no_repeat_ngram_size":no_repeat_ngram_size,
               "eos_token_id": self.tokenizer.eos_token_id})
        # logger.info(f"gpt: out sequences are generated... \n{out_seqs}")
        out_texts = []
        for i, out_seq in enumerate(out_seqs):
            # logger.info(f"out_seq:{out_seq}")
            out_seq = out_seq.tolist()
            out_seq = out_seq[out_seq.index(self.tokenizer.sep_token_id) + 1:]
            # logger.info(f"out_seq_list:{out_seq}")
            out_text = self.tokenizer.decode(out_seq, clean_up_tokenization_spaces=True,skip_special_tokens=True)
            # logger.info(f"out_text:{out_text}")
            out_texts.append(out_text)
        return out_texts

    def server_inference_batch(
            self,
            summaries=None,
            contexts=None,
            lengths=None,
            text_labels=None,
            max_length=512,
            do_sample=True,
            top_k=None,
            top_p=0.9,
            temperature=1,
            no_repeat_ngram_size=0,
            num_return_sequences=1,
            num_beams=1):
        src_texts=[]

        for summary,context,length,text_label in zip(summaries,contexts,lengths,text_labels):
            src_text = self.prepare_inference_input(summary=summary,
                                                    context=context,
                                                    length=length,
                                                    text_label=text_label)
            src_texts.append(src_text)

        #logger.info(f'[inference] src_text:\n{src_texts}')
        inputs = self.encode(src_texts)
        logger.info("==>>> inference in gpt model...")
        out_seqs = self.generate(
            **{"input_ids": inputs["input_ids"].to(self.model.device),
               "attention_mask":inputs['attention_mask'].to(self.model.device),
               "max_length": max_length,
               "do_sample": do_sample,
               "early_stopping": True,
               "num_beams": num_beams,
               "top_k": top_k,
               "top_p": top_p,
               "temperature":temperature,
               # "no_repeat_ngram_size":no_repeat_ngram_size,
               "num_return_sequences":num_return_sequences,
               "eos_token_id": self.tokenizer.eos_token_id})
        # logger.info(f"gpt: out sequences are generated... \n{out_seqs}")
        out_texts = []
        for i, out_seq in enumerate(out_seqs):
            # logger.info(f"out_seq:{out_seq}")
            out_seq = out_seq.tolist()
            out_seq = out_seq[out_seq.index(self.tokenizer.sep_token_id) + 1:]
            # logger.info(f"out_seq_list:{out_seq}")
            out_text = self.tokenizer.decode(out_seq, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            # logger.info(f"out_text:{out_text}")
            out_texts.append(out_text)
        return out_texts

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        self.model.load_state_dict(state_dict)
