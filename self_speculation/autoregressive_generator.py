# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional, Tuple

import torch

import transformers
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
from self_speculation.llama_model_utils import (
    decode_next_token,
    forward,
    forward_early,
)

import torch.nn.functional as F
from confidence_measures import compute_confidence, should_exit

class AutoRegressiveGenerationStrategy(GenerationStrategy):
    def generate_token_ids(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: List[int],
        eos_token_id: int,
        generation_config: GenerationConfig,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None,
    ) -> GenerationStrategyResult:
        """Variant of `generate` with inputs/outputs formatted as token_ids."""
        past_key_values = None

        input_ids: torch.Tensor = torch.tensor([input_ids]).to(model.device)
        output_ids: List[int] = []

        exit_query_cache = None
        for _ in range(generation_config.max_steps):
            if generation_config.exit_layer > 0:
                model_output = forward_early(
                    model,
                    input_ids,
                    past_key_values,
                    generation_config.exit_layer,
                    exit_query_cache,
                )
            else:
                model_output = forward(
                    model,
                    input_ids,
                    past_key_values,
                )
            logits = model_output.logits
            if logits_processors:
                logits = logits_processors(input_ids, logits)
            past_key_values = model_output.past_key_values
            next_token, _ = decode_next_token(logits=logits, token_idx=-1, sample=generation_config.sample, temperature=generation_config.temperature, top_k=generation_config.top_k, top_p=generation_config.top_p)
            if streamer:
                streamer.put(next_token)
            next_token = next_token.item()
            if next_token == eos_token_id:
                break
            if stopping_criteria:
                # TODO: when implementing batch size > 1, stop each sample separately?
                if torch.all(stopping_criteria(input_ids, scores=None)):
                    break
            output_ids.append(next_token)
            # Don't concatenate `next_token` to original `input_ids` since we're using
            # the KV cache (`past_key_values`) to speed up generation.
            input_ids = torch.tensor([[next_token]]).to(input_ids)

        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=None,
        )

class AutoRegressiveGenerationStrategyWithCALM(GenerationStrategy):
    def generate_token_ids(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: List[int],
        eos_token_id: int,
        generation_config: GenerationConfig,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None,
    ) -> GenerationStrategyResult:
        """AutoRegressive strategy with CALM integrated for dynamic exit_layer selection."""
        past_key_values = None
        input_ids_tensor = torch.tensor([input_ids]).to(model.device)  # Shape: [batch_size, seq_length]
        output_ids: List[int] = []
        exit_query_cache = None

        batch_size = input_ids_tensor.size(0)
        prev_hidden_state = torch.zeros(batch_size, model.config.hidden_size).to(model.device)

        accept_count = 0
        total_checks = 0

        for step in range(generation_config.max_steps):
            # Forward pass
            if generation_config.exit_layer > 0:
                model_output = forward_early(
                    model,
                    input_ids_tensor,
                    past_key_values,
                    generation_config.exit_layer,
                    exit_query_cache,
                )
            else:
                model_output = forward(
                    model,
                    input_ids_tensor,
                    past_key_values,
                )
            logits = model_output.logits  # Shape: [batch_size, seq_length, vocab_size]
            if logits_processors:
                logits = logits_processors(input_ids_tensor, logits)
            past_key_values = model_output.past_key_values

            # Decode next token
            next_token, _ = decode_next_token(
                logits=logits,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )
            if streamer:
                streamer.put(next_token)
            next_token_item = next_token.item()
            output_ids.append(next_token_item)

            if next_token_item == eos_token_id:
                break

            # Update input_ids_tensor for next step
            input_ids_tensor = next_token.unsqueeze(0).unsqueeze(0)  # Shape: [batch_size, seq_length=1]

            # Compute confidence
            if model_output.hidden_states is not None and len(model_output.hidden_states) > 0:
                new_state = model_output.hidden_states[-1][:, -1, :]  # [batch_size, hidden_size]
            else:
                new_state = torch.zeros(batch_size, model.config.hidden_size).to(model.device)

            confidence = compute_confidence(
                logits=logits[:, -1, :],  # [batch_size, vocab_size]
                prev_state=prev_hidden_state,
                new_state=new_state,
                conf_method=generation_config.conf_method,
            )
            exit_now = should_exit(confidence, generation_config.conf_threshold)

            # Update prev_hidden_state
            prev_hidden_state = new_state

            # Update acceptance metrics
            accept_count += exit_now.sum().item()
            total_checks += 1

            if exit_now.any():
                # Adjust exit_layer based on confidence
                generation_config.exit_layer = generation_config.min_exit_layer
                # Optionally, you can add a print statement or logging here
                # print(f"Step {step}: High confidence detected. Adjusting exit_layer to {generation_config.min_exit_layer}.")
                continue  # Continue with the adjusted exit_layer in the next iteration

            if stopping_criteria:
                if torch.all(stopping_criteria(input_ids_tensor, scores=None)):
                    break

        acceptance_rate = accept_count / total_checks if total_checks > 0 else 0.0

        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=acceptance_rate,
        )
