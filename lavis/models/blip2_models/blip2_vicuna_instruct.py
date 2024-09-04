"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""

import logging
import string
from operator import is_

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from lavis.common.registry import registry
from lavis.experimental.bipartite_downsampling import semantics_retriever
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    disabled_train,
    episodic_compressor,
    reset_memory_banks,
)
from packaging import version
from torch.cuda.amp import autocast as autocast


@registry.register_model("blip2_vicuna_instruct_hermes")
class Blip2VicunaInstruct_HERMES(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
        memory_bank_length=0,
        num_frames=0,
        window_size=1,
        num_frames_global=0,
        is_zero_shot=False,
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse(
            "4.28"
        ), "BLIP-2 Vicuna requires transformers>=4.28"
        from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
        from transformers import LlamaTokenizer

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token,
            self.visual_encoder.num_features,
            memory_bank_length=memory_bank_length,
            num_frames=num_frames,
        )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(
            llm_model, use_fast=False, truncation_side="left"
        )
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )
        self.llm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llm_tokenizer.add_special_tokens({"bos_token": "</s>"})
        self.llm_tokenizer.add_special_tokens({"eos_token": "</s>"})
        self.llm_tokenizer.add_special_tokens({"unk_token": "</s>"})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input
        self.num_query_token = num_query_token
        self.memory_bank_length = memory_bank_length
        self.num_frames_global = num_frames_global
        self.use_memory_bank = True if memory_bank_length > 0 else False
        self.visual_memory_bank = None

        # TODO: image_pe is 20 for breakfast, num_frames for others
        self.image_pe = nn.Embedding(num_frames, self.visual_encoder.num_features)
        nn.init.constant_(self.image_pe.weight, 0.0)

        self.window_size = window_size
        self.is_zero_shot = is_zero_shot
        self.use_global = True if num_frames_global > 0 else False

        if self.is_zero_shot:
            logging.info("Zero-shot setting")

        if self.use_global and not self.is_zero_shot:
            logging.info("Initializing Hierarchical Q-Former for Global Features")
            # Initialize adapter for frame-to-video Qformer
            self.vqformer_adapter = nn.Linear(
                self.Qformer.config.hidden_size,
                self.visual_encoder.num_features,
            )

            self.frame_Qformer, self.frame_query_tokens = self.init_frame_Qformer(
                num_query_token, self.visual_encoder.num_features
            )

            if not qformer_text_input:
                self.frame_Qformer.bert.embeddings.word_embeddings = None
                self.frame_Qformer.bert.embeddings.position_embeddings = None
                for layer in self.frame_Qformer.bert.encoder.layer:
                    layer.output = None
                    layer.intermediate = None
            else:
                self.frame_Qformer.resize_token_embeddings(len(self.tokenizer))
            self.frame_Qformer.cls = None

            self.video_frame_position_embedding = nn.Embedding(
                self.num_frames_global, self.visual_encoder.num_features
            )
            # nn.init.constant_(self.video_frame_position_embedding.weight, 0.0)

    def encode_global_features(self, images):
        video_encoder_hidden_states = torch.cat(images, dim=1)  # [B, T, N, C]

        B, T, N, C = video_encoder_hidden_states.size()

        # Semantic Knowledge Distillation
        merge, _ = semantics_retriever(
            video_encoder_hidden_states.view(B, T, N * C), T // self.num_frames_global
        )
        video_encoder_hidden_states = merge(
            video_encoder_hidden_states.view(B, T, N * C)
        )

        video_encoder_hidden_states = video_encoder_hidden_states.view(
            B * self.num_frames_global, -1, C
        )

        query_tokens = self.frame_query_tokens.expand(
            video_encoder_hidden_states.size(0), -1, -1
        )
        image_atts = torch.ones(
            video_encoder_hidden_states.size()[:-1], dtype=torch.long
        ).to(video_encoder_hidden_states.device)

        query_output = self.frame_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=video_encoder_hidden_states,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # add frame_pos embedding
        position_ids = torch.arange(
            self.num_frames_global,
            dtype=torch.long,
            device=video_encoder_hidden_states.device,
        )
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        frame_position_embeddings = self.video_frame_position_embedding(position_ids)
        q_hidden_state = query_output.last_hidden_state
        q_hidden_state = self.vqformer_adapter(q_hidden_state)

        frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
        frame_hidden_state = einops.rearrange(
            q_hidden_state, "(b t) q h -> b t q h", b=B, t=self.num_frames_global
        )
        frame_hidden_state = frame_position_embeddings + frame_hidden_state

        # frame attention
        frame_hidden_state = einops.rearrange(
            frame_hidden_state, "b t q h -> b (t q) h", b=B, t=self.num_frames_global
        )

        video_query_tokens = self.query_tokens.expand(
            frame_hidden_state.shape[0], -1, -1
        )

        frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(
            frame_hidden_state.device
        )  # [B, N]
        reset_memory_banks(self.Qformer.bert)
        video_query_output = self.Qformer.bert(
            query_embeds=video_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=frame_atts,
            return_dict=True,
        )
        reset_memory_banks(self.Qformer.bert)

        video_hidden = video_query_output.last_hidden_state
        return video_hidden

    def video_encoder(self, images):
        video_encoder_hidden_states = torch.cat(images, dim=1)  # [B, T, N, C]

        B, T, N, C = video_encoder_hidden_states.size()

        # Semantic Knowledge Distillation
        merge, _ = semantics_retriever(
            video_encoder_hidden_states.view(B, T, N * C), T // self.num_frames_global
        )
        video_encoder_hidden_states = merge(
            video_encoder_hidden_states.view(B, T, N * C)
        )

        # Reshape the output to (Batch, num_frames_1 * num_tokens, channel)
        video_encoder_hidden_states = video_encoder_hidden_states.view(B, -1, C)

        video_query_tokens = self.query_tokens.expand(B, -1, -1)
        video_atts = torch.ones(
            video_encoder_hidden_states.size()[:-1], dtype=torch.long
        ).to(video_encoder_hidden_states.device)

        reset_memory_banks(self.Qformer.bert)
        video_query_output = self.Qformer.bert(
            query_embeds=video_query_tokens,
            encoder_hidden_states=video_encoder_hidden_states,
            encoder_attention_mask=video_atts,
            return_dict=True,
        )
        reset_memory_banks(self.Qformer.bert)

        video_hidden = video_query_output.last_hidden_state
        return video_hidden

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens["input_ids"].append(
                torch.cat(
                    [
                        input_ids[i][:this_input_ones],
                        output_ids[i][1:],
                        input_ids[i][this_input_ones:],
                    ]
                )
            )
            llm_tokens["attention_mask"].append(
                torch.cat(
                    [
                        input_atts[i][:this_input_ones],
                        output_atts[i][1:],
                        input_atts[i][this_input_ones:],
                    ]
                )
            )
        llm_tokens["input_ids"] = torch.stack(llm_tokens["input_ids"])
        llm_tokens["attention_mask"] = torch.stack(llm_tokens["attention_mask"])
        return llm_tokens, input_part_targets_len

    def forward(self, samples):
        image = samples["image"]
        # For video data
        is_video = False
        if image.dim() == 5:
            is_video = True
            B, C, T, H, W = image.shape

        if self.qformer_text_input:
            if is_video:
                text_input = [text for text in samples["text_input"] for _ in range(T)]
            else:
                text_input = samples["text_input"]

            if self.use_memory_bank:
                query_tokens = self.query_tokens.expand(B, -1, -1)  # [B, 32, C]
                text_Qformer = self.tokenizer(
                    samples["text_input"],  # [B]
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                    image.device
                )  # [B, N]
                Qformer_atts = torch.cat(
                    [query_atts, text_Qformer.attention_mask], dim=1
                )

                global_frames = []
                for start in range(0, T, self.window_size):
                    end = min(start + self.window_size, T)
                    window_frames = image[
                        :, :, start:end, :, :
                    ]  # [B, C, self.window_size, H, W]

                    # Process all frames in the window simultaneously
                    B, C, window, H, W = window_frames.shape

                    with self.maybe_autocast():
                        window_frames = window_frames.transpose(1, 2)
                        window_frames = einops.rearrange(
                            window_frames, "b t c h w -> (b t) c h w"
                        )
                        frame_embeds = self.ln_vision(
                            self.visual_encoder(window_frames)
                        )  # [B*self.window_size, 256+1, 1408]

                    N, C = frame_embeds.shape[-2:]
                    frame_embeds = frame_embeds.view(
                        B, -1, N, C
                    )  # [B, self.window_size, 256+1, 1408]

                    position_ids = torch.arange(
                        (end - start), dtype=torch.long, device=frame_embeds.device
                    )  # [window_size]
                    position_ids = position_ids.unsqueeze(0).expand(
                        B, -1
                    )  # [B, window_size]
                    frame_embeds = frame_embeds + self.image_pe(position_ids).unsqueeze(
                        -2
                    )  # [B, window_size, 256+1, 1408]

                    global_frames.append(frame_embeds)

                    if start == 0:
                        encoder_hidden_states = frame_embeds.view(B, -1, C)
                        self.visual_memory_bank = frame_embeds.detach()
                        self.size_constant = torch.ones(B, window, N).to(
                            frame_embeds.device
                        )
                        self.compression_size = torch.ones(
                            B, window, frame_embeds.size(-2)
                        ).to(frame_embeds.device)
                    else:
                        self.visual_memory_bank = torch.cat(
                            [self.visual_memory_bank, frame_embeds.detach()], dim=1
                        )  # Concatenate carry over and new window
                        self.compression_size = torch.cat(
                            [self.compression_size, self.size_constant], dim=1
                        )

                        if self.visual_memory_bank.size(1) > self.memory_bank_length:
                            self.visual_memory_bank, self.compression_size = (
                                episodic_compressor(
                                    self.visual_memory_bank,
                                    self.compression_size,
                                    self.memory_bank_length,
                                )
                            )  # Compress and keep the episodes

                        encoder_hidden_states = self.visual_memory_bank.view(B, -1, C)
                    image_atts = torch.ones(
                        encoder_hidden_states.size()[:-1], dtype=torch.long
                    ).to(
                        image.device
                    )  # [B, N]

                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )

                    # Clear memory bank after processing all windows
                    if end == T:
                        del self.visual_memory_bank
                        del self.compression_size
                        reset_memory_banks(self.Qformer.bert)
            else:
                query_tokens = self.query_tokens.expand(B * T, -1, -1)
                text_Qformer = self.tokenizer(
                    text_input,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                Qformer_atts = torch.cat(
                    [query_atts, text_Qformer.attention_mask], dim=1
                )

                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
        else:
            query_tokens = self.query_tokens.expand(B * T, -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        if self.use_global:
            global_hidden = self.encode_global_features(global_frames)
            inputs_llm = torch.cat(
                [
                    query_output.last_hidden_state[:, : query_tokens.size(1), :],
                    global_hidden,
                ],
                dim=1,
            )
            inputs_llm = self.llm_proj(inputs_llm)
        else:
            inputs_llm = self.llm_proj(
                query_output.last_hidden_state[:, : query_tokens.size(1), :]
            )

        if is_video:
            inputs_llm = inputs_llm.reshape(B, -1, inputs_llm.shape[-1])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"
        text_input_tokens = self.llm_tokenizer(
            samples["text_input"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        self.llm_tokenizer.truncation_side = "right"
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples["text_output"]],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens["input_ids"].masked_fill(
            llm_tokens["input_ids"] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens["input_ids"])
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens["attention_mask"]], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["image"]
        # For video data
        is_video = False
        if image.dim() == 5:
            is_video = True
            B, C, T, H, W = image.shape

        if isinstance(prompt, str):
            prompt = [prompt] * B
        else:
            assert (
                len(prompt) == B
            ), "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [
                p.format(", ".join(samples["ocr_tokens"][i][:30]))
                for i, p in enumerate(prompt)
            ]

        assert self.qformer_text_input == True
        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            if is_video:
                text_input = []
                for text in prompt:
                    text_input.extend([text] * T)
            else:
                text_input = prompt

            if self.use_memory_bank:
                query_tokens = self.query_tokens.expand(B, -1, -1)  # [B, 32, C]
                text_Qformer = self.tokenizer(
                    prompt,  # [B]
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                    image.device
                )  # [B, 32]
                Qformer_atts = torch.cat(
                    [query_atts, text_Qformer.attention_mask], dim=1
                )

                global_frames = []

                for start in range(0, T, self.window_size):
                    end = min(start + self.window_size, T)
                    window_frames = image[
                        :, :, start:end, :, :
                    ]  # [B, C, self.window_size, H, W]

                    # Process all frames in the window simultaneously
                    B, C, window, H, W = window_frames.shape

                    # print(B, C, W, H, W)

                    with self.maybe_autocast():
                        window_frames = window_frames.transpose(1, 2)
                        window_frames = einops.rearrange(
                            window_frames, "b t c h w -> (b t) c h w"
                        )
                        frame_embeds = self.ln_vision(
                            self.visual_encoder(window_frames)
                        ).detach()  # [B*self.window_size, 256+1, 1408]

                    N, C = frame_embeds.shape[-2:]
                    frame_embeds = frame_embeds.view(
                        B, -1, N, C
                    )  # [B, self.window_size, 256+1, 1408]

                    position_ids = torch.arange(
                        (end - start), dtype=torch.long, device=frame_embeds.device
                    )  # [window_size]
                    position_ids = position_ids.unsqueeze(0).expand(
                        B, -1
                    )  # [B, window_size]
                    frame_embeds = frame_embeds + self.image_pe(position_ids).unsqueeze(
                        -2
                    )  # [B, window_size, 256+1, 1408]

                    global_frames.append(frame_embeds)

                    if start == 0:
                        self.visual_memory_bank = frame_embeds
                        self.size_constant = torch.ones(B, window, N).to(
                            frame_embeds.device
                        )
                        self.compression_size = self.size_constant
                    else:
                        self.visual_memory_bank = torch.cat(
                            [self.visual_memory_bank, frame_embeds], dim=1
                        )
                        self.compression_size = torch.cat(
                            [self.compression_size, self.size_constant], dim=1
                        )

                        if self.visual_memory_bank.size(1) > self.memory_bank_length:
                            self.visual_memory_bank, self.compression_size = (
                                episodic_compressor(
                                    self.visual_memory_bank,
                                    self.compression_size,
                                    self.memory_bank_length,
                                )
                            )

                    image_atts = torch.ones(
                        self.visual_memory_bank.view(B, -1, C).size()[:-1],
                        dtype=torch.long,
                    ).to(
                        image.device
                    )  # [B, N]

                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=self.visual_memory_bank.view(B, -1, C),
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )

                    # Clear memory bank after processing all windows
                    if end == T:
                        del self.visual_memory_bank
                        del self.compression_size
                        reset_memory_banks(self.Qformer.bert)

            # TODO: This is not implemented
            else:
                query_tokens = self.query_tokens.expand(B * T, -1, -1)
                text_Qformer = self.tokenizer(
                    text_input,  # [B*T]
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                Qformer_atts = torch.cat(
                    [query_atts, text_Qformer.attention_mask], dim=1
                )
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
        else:
            query_tokens = self.query_tokens.expand(B * T, -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        if self.use_global:
            # For the zero-shot setting, we avoid introducing random weights
            if self.is_zero_shot:
                global_hidden = self.video_encoder(global_frames)
            else:
                global_hidden = self.encode_global_features(global_frames)

            inputs_llm = torch.cat(
                [
                    query_output.last_hidden_state[:, : query_tokens.size(1), :],
                    global_hidden,
                ],
                dim=1,
            )
            inputs_llm = self.llm_proj(inputs_llm)
        else:
            inputs_llm = self.llm_proj(
                query_output.last_hidden_state[:, : query_tokens.size(1), :]
            )

        if is_video:
            inputs_llm = inputs_llm.reshape(B, -1, inputs_llm.shape[-1])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        llm_tokens = self.llm_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                max_new_tokens=max_length,  # TODO: Put this in config
            )

        outputs[outputs < 2] = 2  # convert output id -1/0/1 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs,
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if "ocr_tokens" in samples:
                    text_input = [
                        prompt.format(
                            ", ".join(samples["ocr_tokens"][i][:30]),
                            samples["text_input"][i],
                        )
                        for i in range(len(samples["text_input"]))
                    ]
                elif "choices" in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [
                            f"({string.ascii_lowercase[j]}) {ch}"
                            for j, ch in enumerate(samples["choices"][i])
                        ]
                        this_choices = " ".join(this_choices)
                        text_input.append(
                            prompt.format(samples["text_input"][i], this_choices)
                        )
            else:
                text_input = [
                    prompt.format(question) for question in samples["text_input"]
                ]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty,
            num_captions=num_beams,
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if "context" in samples.keys():
                    this_sample["context"] = [samples["context"][i]]

                if "history" in samples.keys():
                    this_sample["history"] = [samples["history"][i]]

                if "caption" in samples.keys():
                    this_sample["caption"] = [samples["caption"][i]]

                this_result = self._predict_class(
                    this_sample, candidates[i], n_segments
                )
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert (
                len(prompt) == bs
            ), "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [
                    prompt[i].format(*samples["text_input"][i])
                    for i in range(len(prompt))
                ]
            else:
                prompt = [
                    prompt[i].format(samples["text_input"][i])
                    for i in range(len(prompt))
                ]

        # scienceqa
        if "context" in samples.keys() and samples["context"] != "":
            prompt = [
                f'context: {samples["context"][i]}. {prompt[i]}'
                for i in range(len(prompt))
            ]

        # visual dialog
        if "history" in samples.keys() and samples["history"][0] != "":
            prompt = [
                f'dialog history: {samples["history"][i]}\n{prompt[i]}'
                for i in range(len(prompt))
            ]

        if "caption" in samples.keys() and samples["caption"][0] != "":
            prompt = [
                f'This image has the caption "{samples["caption"][i]}". {prompt[i]}'
                for i in range(len(prompt))
            ]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                image.device
            )
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:, :, j, :, :]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(
                        frame_embeds.size()[:-1], dtype=torch.long
                    ).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_llm = self.llm_proj(
                    frame_query_output.last_hidden_state[:, : query_tokens.size(1), :]
                )
                frame_atts_llm = torch.ones(
                    frame_inputs_llm.size()[:-1], dtype=torch.long
                ).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(
                query_output.last_hidden_state[:, : query_tokens.size(1), :]
            )
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(
                image.device
            )

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"
        text_input_tokens = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            # truncation=True,
            # max_length=self.max_txt_len,
        ).to(image.device)

        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )

        # self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "right"
        n_cands = len(candidates)
        with self.maybe_autocast(dtype=torch.bfloat16):
            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len

                this_output_tokens = self.llm_tokenizer(
                    candidates[start_i:end_i],
                    return_tensors="pt",
                    padding="longest",
                    # truncation=True,
                    # max_length=self.max_output_txt_len,
                ).to(image.device)

                this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(
                    seg_len, dim=0
                )
                this_input_tokens_atts = (
                    text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)
                )

                this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
                this_output_tokens_atts = this_output_tokens.attention_mask.repeat(
                    bs, 1
                )

                this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
                    this_input_tokens_ids,
                    this_input_tokens_atts,
                    this_output_tokens_ids,
                    this_output_tokens_atts,
                )

                this_llm_input_ids = this_llm_tokens["input_ids"]
                this_llm_atts = this_llm_tokens["attention_mask"]
                # this_llm_input_ids = torch.cat([this_input_tokens_ids, this_output_tokens_ids], dim=1)
                # this_llm_atts = torch.cat([this_input_tokens_atts, this_output_tokens_atts], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(
                    this_llm_input_ids
                )
                inputs_embeds = torch.cat(
                    [inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1
                )
                attention_mask = torch.cat(
                    [atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1
                )

                this_targets = this_llm_input_ids.masked_fill(
                    this_llm_input_ids == self.llm_tokenizer.pad_token_id, -100
                )
                # this_targets[:, :this_input_tokens_ids.size(1)] = -100
                for i, l in enumerate(this_input_targets_len):
                    this_targets[i][:l] = -100

                this_targets = torch.cat(
                    [empty_targets.repeat_interleave(seg_len, dim=0), this_targets],
                    dim=1,
                )

                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )

                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)

        return output_class_ranks

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")
        memory_bank_length = cfg.get("memory_bank_length", 0)
        num_frames = cfg.get("num_frames", 0)
        window_size = cfg.get("window_size", 1)
        num_frames_global = cfg.get("num_frames_global", 0)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)

        is_zero_shot = cfg.get("is_zero_shot", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            memory_bank_length=memory_bank_length,
            num_frames=num_frames,
            window_size=window_size,
            num_frames_global=num_frames_global,
            is_zero_shot=is_zero_shot,
        )
        model.load_checkpoint_from_config(cfg)

        return model
