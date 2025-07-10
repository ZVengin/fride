import random

import torch
import torch.nn as nn
from typing import Optional

from torch.nn import CrossEntropyLoss
from transformers import BartTokenizer, BartModel, BartConfig, BartPretrainedModel
from transformers.models.bart.modeling_bart import BartDecoder, _expand_mask
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput, \
    BaseModelOutputWithPastAndCrossAttentions
from utils import logger, max_summary, max_target, max_context, assign_label, MODES

BART_MAX_LEN = 1024


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
    """
    prev_output_tokens = input_ids.clone()

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    """Bart use the eos_token as the start token"""
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens

class BART(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: BartConfig):
        super(BART, self).__init__(config)
        self.model = BartModel(config)
        #self.model.decoder = MyBartDecoder(self.model.config, self.model.shared)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.tokenizer = None
        self.first_device = None
        self.second_device = None
        self.add_length = None
        self.add_mode = None
        self.add_summary = None

        self.init_weights()

    def set_input(self,add_mode,add_length, add_summary):
        self.add_mode=add_mode
        self.add_length=add_length
        self.add_summary=add_summary

    def freeze_parameters(self,freeze=False):
        for p in self.model.encoder.parameters():
            p.requires_grad = not freeze
        #for p in self.model.shared.parameters():
        #    p.requires_grad = not freeze

    def split_to_gpus(self, device_map=None):
        if device_map is None:
            return
        #elif len(device_map) < 2:
        #    device_count = 1
        #else:
        #    device_count = 2
        self.first_device = torch.device(device_map['encoder'])
        self.second_device = torch.device(device_map['decoder']) #if device_count > 1 else torch.device("cuda:0")
        # self.model.shared = self.model.shared.to(self.second_device)

        # print(self.model.shared.weight.device)

        # self.model.encoder.embed_positions = self.model.encoder.embed_positions.to(self.first_device)
        # self.model.encoder.layers = self.model.encoder.layers.to(self.first_device)
        self.model.encoder = self.model.encoder.to(self.first_device)
        # print(self.model.shared.weight.device)

        # self.model.decoder.embed_positions = self.model.decoder.embed_positions.to(self.second_device)
        # self.model.decoder.layers = self.model.decoder.layers.to(self.second_device)
        self.model.decoder = self.model.decoder.to(self.second_device)
        # print(self.model.shared.weight.device)

        self.lm_head = self.lm_head.to(self.second_device)
        # print(self.model.shared.weight.device)
        self.final_logits_bias = self.final_logits_bias.to(self.second_device)
        # print(self.model.shared.weight.device)

        if self.model.shared is None:
            self.model.encoder.embed_tokens = self.model.encoder.embed_tokens.to(self.first_device)
            self.model.decoder.embed_tokens = self.model.decoder.embed_tokens.to(self.second_device)
        # print(self.model.shared.weight.device)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                target_length=None,
                decoder_attention_mask=None,
                #head_mask=None,
                #decoder_head_mask=None,
                #cross_attn_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None, ):
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
                decoder_input_ids = decoder_input_ids.to(self.second_device)

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
            decoder_input_ids = decoder_input_ids.to(self.second_device)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = input_ids.to(self.second_device)
            inputs_embeds = self.model.encoder.embed_tokens(input_ids) * self.model.encoder.embed_scale
            inputs_embeds = inputs_embeds.to(self.first_device)
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds

        if inputs_embeds is None and encoder_outputs is None:
            raise ValueError("You should specify either input_ids, inputs_embeds or encoder outputs")

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.first_device)

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        encoder_hidden_states = encoder_outputs[0].to(self.second_device)
        attention_mask = attention_mask.to(self.second_device) if attention_mask is not None else attention_mask

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            #target_length=target_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(self.second_device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, auto_decoder_input_ids,decoder_input_ids=None, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, target_length=None,**kwargs
    ):
        # cut decoder_input_ids if past is used
        decoder_input_ids = auto_decoder_input_ids
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "target_length":target_length,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def encode(self, text):
        #BART tokenizer will add special tokens <s> </s> at the start and the end of the text
        inputs = self.tokenizer(text, truncation=True,padding=True,max_length=BART_MAX_LEN, return_tensors="pt")
        # inputs = {k:v.to(self.model.shared.weight.device) for k,v in inputs.items()}
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
            target = f"{target}{self.tokenizer.eos_token * 30}"
            tgt_text = self.tokenizer.decode(
                self.tokenizer.encode(target, add_special_tokens=False)[:(target_cluster_label + 1) * 20])
        else:
            tgt_text = f"{target} {self.tokenizer.eos_token}"

        src_text = (f"{sample.target_label} {self.tokenizer.additional_special_tokens[0]} "
                    if self.add_mode else "")\
                   + (f"{target_cluster_label} {self.tokenizer.additional_special_tokens[1]} "
                      if self.add_length else "") \
                   + (f"{summary} {self.tokenizer.additional_special_tokens[2]} "
                      if self.add_summary else "")\
                   + f"{context}"

        tgt_length = len(self.tokenizer.tokenize(tgt_text))
        return src_text, tgt_text, tgt_length

    def get_loss(self, batch):
        src_txts,tgt_txts,tgt_lengths = [],[],[]
        for sample in batch:
            src_text, tgt_text, tgt_length = self.prepare_inputs(sample)
            src_txts.append(src_text)
            tgt_txts.append(tgt_text)
            tgt_lengths.append(tgt_length)
        #logger.info(f'[train] src_text:\n{src_text}\ntgt_text:{tgt_text}')
        encoder_inputs = self.encode(src_txts)
        decoder_inputs = self.encode(tgt_txts)
        """ we add eos_token to the target text"""

        #decoder_inputs["input_ids"] = decoder_inputs["input_ids"][decoder_inputs["input_ids"]==self.tokenizer.pad_token_id]

        outputs = self.forward(**{
            "input_ids": encoder_inputs["input_ids"],
            "attention_mask": encoder_inputs["attention_mask"],
            "labels": decoder_inputs["input_ids"][:,1:],
            "decoder_attention_mask":decoder_inputs["attention_mask"][:,1:].to(self.model.decoder.device)
            #"target_length": tgt_length
        })
        return outputs.loss

    def load_tokenizer(self, path, cache_dir=None):
        self.tokenizer = BartTokenizer.from_pretrained(path,cache_dir=cache_dir)

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)

    def add_special_tokens(self):
        assert self.tokenizer is not None
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": ["<sep_0>", "<sep_1>", "<sep_2>","<newline>"]
        })
        logger.info(f"{len(self.tokenizer.additional_special_tokens)} tokens is added to the vocabulary...")
        logger.info(f"added additional special tokens:{self.tokenizer.additional_special_tokens}")
        self.resize_token_embeddings(len(self.tokenizer))


    def prepare_inference_input(self,
                                summary=None,
                                context=None,
                                length=None,
                                text_label=None):
        context = self.tokenizer.decode(self.tokenizer.encode(context, add_special_tokens=False)[-max_context:])
        if self.add_summary:
            summary = self.tokenizer.decode(self.tokenizer.encode(summary, add_special_tokens=False)[:max_summary])
        target_cluster_label = length

        src_text = (f"{text_label} {self.tokenizer.additional_special_tokens[0]} "
                      if self.add_mode else "") \
                   + (f"{target_cluster_label} {self.tokenizer.additional_special_tokens[1]} "
                      if self.add_length else "") \
                   + (f"{summary} {self.tokenizer.additional_special_tokens[2]} "
                      if self.add_summary else "") \
                   + f"{context}"
        logger.info(f'src_text in inference of bart model:{src_text}')
        return src_text


    def server_inference(self,
                         summary=None,
                         context=None,
                         length=None,
                         text_label=None,
                         max_length=300,
                         do_sample=True,
                         top_k=None,
                         top_p=0.9,
                         no_repeat_ngram_size=0,
                         num_return_sequences=1,
                         num_beams=1,
                         skip_special_tokens=True):
        src_text = self.prepare_inference_input(summary=summary,
                                                context=context,
                                                length=length,
                                                text_label=text_label)
        #logger.info(f'[inference] src_text:\n{src_text}')
        encoder_inputs = self.encode(src_text)
        logger.info("==>>> inference in bart model...")
        out_seqs = self.generate(
            **{"input_ids": encoder_inputs["input_ids"].to(self.model.device),
               "max_length": max_length,
               "do_sample": do_sample,
               "early_stopping": True,
               "num_beams": num_beams,
               "top_k": top_k,
               "top_p": top_p,
               # "no_repeat_ngram_size":no_repeat_ngram_size,
               "num_return_sequences":num_return_sequences,
               "eos_token_id": self.tokenizer.eos_token_id}
        )
        logger.info("bart: out sequences are generated... ")
        out_texts = []
        for i, out_seq in enumerate(out_seqs):
            out_seq = out_seq.tolist()
            out_text = self.tokenizer.decode(out_seq,
                                             clean_up_tokenization_spaces=True,
                                             skip_special_tokens=skip_special_tokens)
            out_texts.append(out_text)
        return out_texts


    def server_inference_batch(
            self,
            summaries=None,
            contexts=None,
            lengths=None,
            text_labels=None,
            max_length=300,
            do_sample=True,
            top_k=None,
            top_p=0.9,
            temperature=1,
            no_repeat_ngram_size=0,
            num_return_sequences=1,
            num_beams=1,
            skip_special_tokens=True):
        src_texts = []
        for summary,context,length,text_label in zip(summaries,contexts,lengths,text_labels):
            src_text = self.prepare_inference_input(
                summary=summary,
                context=context,
                length=length,
                text_label=text_label)
            src_texts.append(src_text)
        encoder_inputs = self.encode(src_texts)
        logger.info("==>>> inference in bart model...")
        out_seqs = self.generate(
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
               "num_return_sequences":num_return_sequences,
               "eos_token_id": self.tokenizer.eos_token_id}
        )
        logger.info("bart: out sequences are generated... ")
        out_texts = []
        for i, out_seq in enumerate(out_seqs):
            out_seq = out_seq.tolist()
            out_text = self.tokenizer.decode(out_seq,
                                             clean_up_tokenization_spaces=True,
                                             skip_special_tokens=skip_special_tokens)
            out_texts.append(out_text)
        return out_texts
