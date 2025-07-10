import torch
import torch.nn as nn
from utils import logger, assign_label, max_summary, max_target, max_context, MODES
from transformers import T5Tokenizer, T5ForConditionalGeneration

T5_MAX_LEN = 1024


class T5Model(nn.Module):
    def __init__(self):
        super(T5Model, self).__init__()
        self.model = None
        self.tokenizer = None
        self.add_length=None
        self.add_mode=None
        self.add_summary=None

    def set_input(self,add_mode,add_length,add_summary):
        self.add_mode = add_mode
        self.add_length = add_length
        self.add_summary = add_summary

    def split_to_gpus(self, mapping=None):
        if mapping is None:
            return
        self.model.parallelize(mapping)

    def load_tokenizer(self, path, cache_dir=None):
        self.tokenizer = T5Tokenizer.from_pretrained(path, cache_dir=cache_dir)

    def encode(self, text):
        # T5 tokenizer only add special token </s> with id 1 at the end of text
        inputs = self.tokenizer(text,truncation=True, padding=True,max_length=T5_MAX_LEN,
                                return_tensors="pt")
        return inputs

    def decode(self, token_ids):
        text = self.tokenizer.decode(token_ids)
        return text

    def prepare_inputs(self, sample):
        """truncate the context, summary and target"""
        # target_text_label_ids = self.tokenizer.encode(sample.target_label,add_special_tokens=False)
        context = self.tokenizer.decode(self.tokenizer.encode(sample.context, add_special_tokens=False)[-max_context:])
        if self.add_summary:
            summary = self.tokenizer.decode(self.tokenizer.encode(sample.summary, add_special_tokens=False)[:max_summary])
        target_token_ids = self.tokenizer.encode(sample.target, add_special_tokens=False)[:max_target]
        target = self.tokenizer.decode(target_token_ids)

        if self.add_length:
            target_cluster_label = assign_label(len(target_token_ids))
        else:
            target_cluster_label = None

        src_text = (f"writing mode: {sample.target_label} " #{self.tokenizer.additional_special_tokens[0]} "
                      if self.add_mode else "") \
                   + (f"length: {target_cluster_label} " #{self.tokenizer.additional_special_tokens[1]} "
                      if self.add_length else "") \
                   + (f"summary: {summary} "
                      if self.add_summary else "")\
                   + f"context: {context}"

        if self.add_length:
            target = f"{target}{self.tokenizer.eos_token * 30}"
            tgt_text = self.tokenizer.decode(self.tokenizer.encode(target, add_special_tokens=False)[:(target_cluster_label + 1) * 20])
        else:
            tgt_text = f"{target}{self.tokenizer.eos_token}"

        return src_text, tgt_text, target_cluster_label

    def get_loss(self, batch):
        src_texts, tgt_texts = [],[]
        for sample in batch:
            src_text, tgt_text, tgt_length = self.prepare_inputs(sample)
            src_texts.append(src_text)
            tgt_texts.append(tgt_text)
        logger.info(f'[train] src_text:\n{src_text}\ntgt_text:{tgt_text}')
        encoder_inputs = self.encode(src_texts)
        decoder_inputs = self.encode(tgt_texts)
        """ we add eos_token to the target text"""

        #logger.info(f'pad_src_text_length:{len(self.tokenizer.tokenize(tgt_text))},\n'
        #            f'expect_src_text_length:{(tgt_length + 1) * 20}')

        outputs = self.forward({
            "input_ids": encoder_inputs["input_ids"].to(self.model.encoder.embed_tokens.weight.device),
            "attention_mask": encoder_inputs["attention_mask"].to(self.model.encoder.embed_tokens.weight.device),
            "labels": decoder_inputs["input_ids"].to(self.model.lm_head.weight.device),
            "decoder_attention_mask":decoder_inputs["attention_mask"].to(self.model.decoder.embed_tokens.weight.device)
        })
        return outputs.loss

    def forward(self, inputs):
        return self.model.forward(**inputs)

    def generate(self, **inputs):
        return self.model.generate(**inputs)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    @classmethod
    def from_pretrained(cls, path, cache_dir=None):
        t5 = T5ForConditionalGeneration.from_pretrained(path, cache_dir=cache_dir)
        model = T5Model()
        model.model = t5
        return model

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
            "additional_special_tokens": ["<newline>"]
            #"additional_special_tokens": ["<sep_0>", "<sep_1>", "<sep_2>", "<newline>"]
        })
        logger.info(f"{len(self.tokenizer.additional_special_tokens)} tokens is added to the vocabulary...")
        logger.info(f"added additional special tokens:{self.tokenizer.additional_special_tokens}")
        self.model.resize_token_embeddings(len(self.tokenizer))


    def prepare_inference_input(self,
                                summary=None,
                                context=None,
                                length=None,
                                text_label=None):
        context = self.tokenizer.decode(self.tokenizer.encode(context, add_special_tokens=False)[-max_context:])
        if self.add_summary:
            summary = self.tokenizer.decode(self.tokenizer.encode(summary, add_special_tokens=False)[:max_summary])

        target_cluster_label = length
        target_label = text_label

        src_text = (f"writing mode: {target_label} "  # {self.tokenizer.additional_special_tokens[0]} "
                    if self.add_mode else "") \
                   + (f"length: {target_cluster_label} "  # {self.tokenizer.additional_special_tokens[1]} "
                      if self.add_length else "") \
                   + (f"summary: {summary} "
                      if self.add_summary else "")\
                   + f"context: {context}"  # {self.tokenizer.additional_special_tokens[2]} "
        return src_text


    def server_inference(self,
                         summary,
                         context,
                         length,
                         text_label=None,
                         max_length=300,
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
        encoder_inputs = self.encode(src_text)
        logger.info("==>>> inference in T5 model...")
        out_seqs = self.model.generate(
            **{"input_ids": encoder_inputs["input_ids"].to(self.model.device),
               "max_length": max_length,
               "do_sample": do_sample,
               "early_stopping": True,
               "num_beams": num_beams,
               "top_k": top_k,
               "top_p": top_p,
               # "no_repeat_ngram_size":no_repeat_ngram_size,
               "eos_token_id": self.tokenizer.eos_token_id}
        )
        logger.info("t5: out sequences are generated... ")
        out_texts = []
        for i, out_seq in enumerate(out_seqs):
            out_seq = out_seq.tolist()
            out_text = self.tokenizer.decode(out_seq, clean_up_tokenization_spaces=True,skip_special_tokens=True)
            out_texts.append(out_text)
            # logger.info(f"==>>>T5 out text:\n{out_text}")
        return out_texts


    def server_inference_batch(self,
                               summaries,
                               contexts,
                               lengths,
                               text_labels=None,
                               max_length=300,
                               do_sample=True,
                               top_k=None,
                               top_p=0.9,
                               temperature=1,
                               no_repeat_ngram_size=0,
                               num_return_sequences=1,
                               num_beams=1):
        src_texts = []
        for summary,context,length,text_label in zip(summaries,contexts,lengths,text_labels):
            src_text = self.prepare_inference_input(summary=summary,
                                                    context=context,
                                                    length=length,
                                                    text_label=text_label)
            src_texts.append(src_text)
        encoder_inputs = self.encode(src_texts)
        logger.info("==>>> inference in T5 model...")
        out_seqs = self.model.generate(
            **{"input_ids": encoder_inputs["input_ids"].to(self.model.device),
               "attention_mask":encoder_inputs["attention_mask"].to(self.model.device),
               "max_length": max_length,
               "do_sample": do_sample,
               "early_stopping": True,
               "num_beams": num_beams,
               "top_k": top_k,
               "top_p": top_p,
               "temperature":temperature,
               # "no_repeat_ngram_size":no_repeat_ngram_size,
               "num_return_sequences": num_return_sequences,
               "eos_token_id": self.tokenizer.eos_token_id}
        )
        logger.info("t5: out sequences are generated... ")
        out_texts = []
        for i, out_seq in enumerate(out_seqs):
            out_seq = out_seq.tolist()
            out_text = self.tokenizer.decode(out_seq, clean_up_tokenization_spaces=True,skip_special_tokens=True)
            out_texts.append(out_text)
            # logger.info(f"==>>>T5 out text:\n{out_text}")
        return out_texts

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        self.model.load_state_dict(state_dict)
