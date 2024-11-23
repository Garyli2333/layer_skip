# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional, Tuple

import colorama
import torch

import transformers
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
import torch.nn.functional as F
from confidence_measures import compute_confidence, should_exit

from self_speculation.speculative_streamer import SpeculativeTextStreamer
from self_speculation.llama_model_utils import (
    crop_past_key_values,
    decode_next_token,
    forward_early,
    forward_remainder,
)

def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)

class SelfSpeculativeGenerationStrategy(GenerationStrategy):
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
        past_key_values = None

        input_ids_list = input_ids
        input_ids: torch.Tensor = torch.tensor([input_ids_list]).to(model.device)
        output_ids: List[int] = []

        calls: int = 0
        total_draft_matches = 0
        total_generations = 0
        ## use single_step_speculation to generate multiple steps; the result are used to decided whether to stop;
        while len(output_ids) < generation_config.max_steps:
            (
                input_ids,
                output_ids,
                past_key_values,
                number_of_matches,
                num_speculations,
            ) = self.single_step_speculation(
                model=model,
                input_ids_list=input_ids_list,
                input_ids=input_ids,
                output_ids=output_ids,
                num_speculations=min(
                    generation_config.num_speculations,
                    generation_config.max_steps - len(output_ids) - 1,
                ),
                past_key_values=past_key_values,
                exit_layer=generation_config.exit_layer,
                eos_token_id=eos_token_id,
                calls=calls,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                logits_processors=logits_processors,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )
            calls += 1
            total_draft_matches += number_of_matches
            total_generations += num_speculations
            eos_found = False
            if eos_token_id in output_ids:
                # break out of loop when we get an EOS token
                # remove the EOS token id
                output_ids = output_ids[: output_ids.index(eos_token_id)]
                eos_found = True
            if eos_found:
                break
            if stopping_criteria:
                # TODO: when implementing batch size > 1, stop each sample separately?
                if torch.all(stopping_criteria(input_ids, scores=None)):
                    break
        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=total_draft_matches / total_generations,
        )

    # TODO: remove calls, input_ids_list, rely on generation config
    def single_step_speculation(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: torch.Tensor,
        input_ids_list: List[int],
        output_ids: List[int],
        num_speculations: int,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        eos_token_id: int,
        calls: int,
        exit_layer: int,
        sample: Optional[bool] = False,
        temperature: Optional[float] = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None
    ):
        prompt_length: int = input_ids.size(1)
        draft_input_ids = input_ids.clone()
        draft_output_ids: List[int] = []
        if sample:
            draft_probabilities: List[torch.Tensor] = []
        exit_query_cache = None
        for _ in range(num_speculations):
            draft_result = forward_early(
                model,
                draft_input_ids,
                past_key_values,
                exit_layer,
                exit_query_cache,
            )
            past_key_values = draft_result.past_key_values
            exit_query_cache = draft_result.exit_query_cache
            draft_logits = draft_result.logits
            if logits_processors:
                draft_logits = logits_processors(draft_input_ids, draft_logits)
            draft_next_token, draft_next_prob = decode_next_token(logits=draft_logits, token_idx=-1, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
            draft_next_token = draft_next_token.item()
            draft_output_ids.append(draft_next_token)
            if sample:
                draft_probabilities.append(draft_next_prob)
            draft_input_ids = torch.tensor([[draft_next_token]]).to(draft_input_ids)
            if draft_next_token == eos_token_id:
                # break out of loop when we get an EOS token
                break

        # input_ids (1 x T_p) and draft_output_ids (1 x T_d) are concatenated together to make
        # 1 x (T_d  + T_p)
        draft_output_ids = torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids)
        prefill_token_ids = torch.cat(
            [input_ids, draft_output_ids],
            dim=-1,
        )

        if streamer:
            if isinstance(streamer, SpeculativeTextStreamer):
                print(colorama.Fore.LIGHTMAGENTA_EX, end="")
                streamer.put(draft_output_ids, is_draft=True)

        # logits: 1 x (T_d  + T_p) x V
        verify_results = forward_remainder(
            model,
            prefill_token_ids.int(),
            past_key_values,
            exit_layer,
            exit_query_cache,
        )
        logits = verify_results.logits
        if logits_processors:
            logits = logits_processors(prefill_token_ids, logits)
        past_key_values = verify_results.past_key_values
        # only select the logits relevant to what the draft has outputted.
        # verification_logits: 1 x T_d x V
        verification_logits = logits[:, prompt_length - 1 :, :]

        # verified_tokens: 1 x (T_d)
        # There is a predicted token for every token in the draft output ids list, however note that the
        # first tokens (or first N tokens) are coming from the prompt
        verified_tokens, verified_probabilities = decode_next_token(logits=verification_logits, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)

        # skip verification of the last token as it is a new token predicted from the main model
        verified_tokens = verified_tokens.to(prefill_token_ids)
        verified = draft_output_ids[:, :] == verified_tokens[:, :-1]

        # number of matches is the index of the number of tokens we are accepting from the draft
        if not sample:
            number_of_matches = ((~(verified)).cumsum(dim=-1) < 1).sum().item()
        else:
            number_of_matches = 0
            rand = torch.rand_like(draft_output_ids, dtype=torch.float)
            for i in range(draft_output_ids.numel()):
                if rand[0, i] < min(1, verified_probabilities[i, draft_output_ids[0, i]].item() / draft_probabilities[i][0, draft_output_ids[0, i]].item()):
                    number_of_matches += 1
                else:
                    verified_tokens[0][number_of_matches] = torch.multinomial(max_fn((verified_probabilities[i, :] - draft_probabilities[i])), num_samples=1).item()
                    break

        # accept the `number_of_matches` tokens from the draft with one more from the main model
        # since we re-use the same cachem the input id should only be the last accepted token TODO check this
        input_ids = verified_tokens[:, number_of_matches : number_of_matches + 1]
        output_ids.extend(draft_output_ids[0, : number_of_matches].tolist())
        output_ids.extend(verified_tokens[0][number_of_matches : number_of_matches + 1].tolist())

        if streamer:
            if isinstance(streamer, SpeculativeTextStreamer):
                streamer.delete(len(draft_output_ids[0, :]))
                print(colorama.Fore.GREEN, end="")
                streamer.put(draft_output_ids[0, : number_of_matches])
                print(colorama.Style.RESET_ALL, end="")
                streamer.put(verified_tokens[0][number_of_matches : number_of_matches + 1])
            else:
                # streamer.put(torch.cat((draft_output_ids[0, : number_of_matches], verified_tokens[0][number_of_matches : number_of_matches + 1])))
                streamer.put(torch.LongTensor(output_ids[len(output_ids)-number_of_matches-1:]))

        # we want the entire output sequence + input sequence
        past_key_values = crop_past_key_values(
            past_key_values, len(input_ids_list) + len(output_ids) - 1
        )

        return (
            input_ids,
            output_ids,
            past_key_values,
            number_of_matches,
            draft_output_ids.numel(),
        )


class SelfSpeculativeGenerationStrategyWithCALM(GenerationStrategy):
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
        past_key_values = None
        input_ids_tensor = torch.tensor([input_ids]).to(model.device)
        output_ids: List[int] = []

        batch_size = input_ids_tensor.size(0)
        current_tokens = input_ids_tensor[:, -1]  # [batch_size]
        prev_hidden_state = torch.zeros(batch_size, model.config.hidden_size).to(input_ids_tensor.device)

        accept_count = 0
        total_checks = 0

        while len(output_ids) < generation_config.max_steps:
            (
                input_ids,
                output_ids,
                past_key_values,
                hidden_states_collected,
                logits,
                number_of_matches,
                num_speculations,
            ) = self.single_step_speculation(
                model=model,
                input_ids=input_ids_tensor,
                output_ids=output_ids,
                num_speculations=min(
                    generation_config.num_speculations,
                    generation_config.max_steps - len(output_ids) - 1,
                ),
                past_key_values=past_key_values,
                eos_token_id=eos_token_id,
                generation_config=generation_config,
            )

            accept_count += number_of_matches
            total_checks += num_speculations

            if hidden_states_collected:
                new_state = hidden_states_collected[-1][:, -1, :]
            else:
                new_state = torch.zeros(batch_size, model.config.hidden_size).to(input_ids_tensor.device)

            confidence = compute_confidence(
                logits=logits[:, -1, :],
                prev_state=prev_hidden_state,
                new_state=new_state,
                conf_method=generation_config.conf_method
            )
            prev_hidden_state = new_state

            exit_now = should_exit(confidence, generation_config.conf_threshold)  # [batch_size]
            accept_count += exit_now.sum().item()
            total_checks += 1

            if exit_now.any():
                generation_config.exit_layer = generation_config.min_exit_layer

            if eos_token_id in output_ids:
                output_ids = output_ids[:output_ids.index(eos_token_id)]
                break

            if stopping_criteria:
                if torch.all(stopping_criteria(input_ids, scores=None)):
                    break

        acceptance_rate = accept_count / total_checks if total_checks > 0 else 0.0

        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=acceptance_rate,
        )


    def single_step_speculation(
            self,
            model: transformers.LlamaForCausalLM,
            input_ids: torch.Tensor,
            output_ids: List[int],
            num_speculations: int,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
            eos_token_id: int,
            generation_config: GenerationConfig,
    ) -> Tuple[torch.Tensor, List[int], Optional[Tuple[torch.Tensor]], int, int]:
        prompt_length: int = input_ids.size(1)
        draft_input_ids = input_ids.clone()
        draft_output_ids: List[int] = []
        if generation_config.sample:
            draft_probabilities: List[torch.Tensor] = []
        exit_query_cache = None
        hidden_states_collected: List[torch.Tensor] = []

        number_of_matches = 0
        draft_num = 0

        for _ in range(num_speculations):
            # Draft Step
            draft_result = forward_early(
                model,
                draft_input_ids,
                past_key_values,
                generation_config.exit_layer,
                exit_query_cache,
            )
            past_key_values = draft_result.past_key_values
            exit_query_cache = draft_result.exit_query_cache
            draft_logits = draft_result.logits
            hidden_states_collected.extend(draft_result.hidden_states) # @ gary

            if generation_config.logits_processors:
                draft_logits = generation_config.logits_processors(draft_input_ids, draft_logits)

            draft_next_token, draft_next_prob = decode_next_token(
                logits=draft_logits,
                token_idx=-1,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p
            )
            draft_next_token = draft_next_token.item()
            draft_output_ids.append(draft_next_token)

            if generation_config.sample:
                draft_probabilities.append(draft_next_prob)

            draft_input_ids = torch.tensor([[draft_next_token]]).to(draft_input_ids.device)

            if draft_next_token == eos_token_id:
                break

        # Concatenate draft tokens with input
        draft_output_ids_tensor = torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids.device)
        prefill_token_ids = torch.cat([input_ids, draft_output_ids_tensor], dim=-1)

        # Streamer handling
        if generation_config.streamer and isinstance(generation_config.streamer, SpeculativeTextStreamer):
            print(colorama.Fore.LIGHTMAGENTA_EX, end="")
            generation_config.streamer.put(draft_output_ids, is_draft=True)

        # Verification Step
        verify_results = forward_remainder(
            model,
            prefill_token_ids.int(),
            past_key_values,
            generation_config.exit_layer,
            exit_query_cache,
        )
        logits = verify_results.logits
        hidden_states_collected.extend(verify_results.hidden_states)

        if generation_config.logits_processors:
            logits = generation_config.logits_processors(prefill_token_ids, logits)

        past_key_values = verify_results.past_key_values

        # Extract verification logits
        verification_logits = logits[:, prompt_length - 1:, :]

        # Decode verified tokens
        verified_tokens, verified_probabilities = decode_next_token(
            logits=verification_logits,
            sample=generation_config.sample,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p
        )

        verified_tokens = verified_tokens.to(prefill_token_ids.device)
        verified = draft_output_ids_tensor[:, :] == verified_tokens[:, :-1]

        if not generation_config.sample:
            number_of_matches = ((~verified).cumsum(dim=-1) < 1).sum().item()
        else:
            number_of_matches = 0
            rand = torch.rand_like(draft_output_ids_tensor, dtype=torch.float)
            for i in range(draft_output_ids_tensor.numel()):
                if rand[0, i] < min(1, verified_probabilities[i, draft_output_ids_tensor[0, i]].item() /
                                       draft_probabilities[i][0, draft_output_ids_tensor[0, i]].item()):
                    number_of_matches += 1
                else:
                    verified_tokens[0][number_of_matches] = torch.multinomial(
                        max_fn((verified_probabilities[i, :] - draft_probabilities[i])), num_samples=1).item()
                    break

        # Update output_ids and input_ids
        input_ids = verified_tokens[:, number_of_matches:number_of_matches + 1]
        output_ids.extend(draft_output_ids_tensor[0, :number_of_matches].tolist())
        output_ids.extend(verified_tokens[0, number_of_matches:number_of_matches + 1].tolist())

        # Streamer handling after verification
        if generation_config.streamer:
            if isinstance(generation_config.streamer, SpeculativeTextStreamer):
                generation_config.streamer.delete(len(draft_output_ids_tensor[0, :]))
                print(colorama.Fore.GREEN, end="")
                generation_config.streamer.put(draft_output_ids_tensor[0, :number_of_matches])
                print(colorama.Style.RESET_ALL, end="")
                generation_config.streamer.put(verified_tokens[0, number_of_matches:number_of_matches + 1])
            else:
                generation_config.streamer.put(torch.LongTensor(output_ids[-(number_of_matches + 1):]))

        # Crop past_key_values to manage memory
        past_key_values = crop_past_key_values(
            past_key_values, len(input_ids) + len(output_ids) - 1
        )

        if verify_results.hidden_states is not None and len(verify_results.hidden_states) > 0:
            new_state = verify_results.hidden_states[-1][:, -1, :]
        else:
            new_state = None

        return (
            input_ids,
            output_ids,
            past_key_values,
            hidden_states_collected,
            logits,
            number_of_matches,
            draft_output_ids_tensor.numel(),
        )


class GenerationStrategyWithCALM(GenerationStrategy):
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
        past_key_values = None
        input_ids_tensor = torch.tensor([input_ids]).to(model.device)
        output_ids: List[int] = []

        batch_size = input_ids_tensor.size(0)
        current_tokens = input_ids_tensor[:, -1]  # [batch_size]
        prev_hidden_state = torch.zeros(batch_size, model.config.hidden_size).to(input_ids_tensor.device)

        accept_count = 0
        total_checks = 0

        for step in range(generation_config.max_steps):
            (
                next_token,
                output_ids_step,
                past_key_values,
                new_hidden_state,
                number_of_matches,
                draft_num,
            ) = self.single_step_speculation(
                model=model,
                current_tokens=current_tokens,
                past_key_values=past_key_values,
                prev_hidden_state=prev_hidden_state,
                generation_config=generation_config,
            )

            confidence = compute_confidence(
                logits=logits,
                prev_state=prev_hidden_state,
                new_state=new_hidden_state,
                conf_method=generation_config.conf_method
            )

            prev_hidden_state = new_hidden_state

            exit_now = should_exit(confidence, generation_config.conf_threshold)  # [batch_size]
            accept_count += exit_now.sum().item()
            total_checks += 1

            output_ids.extend(output_ids_step)
            current_tokens = next_token.unsqueeze(0)

            if next_token.item() == eos_token_id:
                break

            if (step + 1) % generation_config.exit_interval != 0:
                continue

        acceptance_rate = accept_count / total_checks if total_checks > 0 else 0.0

        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=acceptance_rate
        )

    def single_step_speculation(
        self,
        model: transformers.LlamaForCausalLM,
        current_tokens: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor]],
        prev_hidden_state: torch.Tensor,
        generation_config: GenerationConfig,
    ) -> Tuple[torch.Tensor, List[int], Optional[Tuple[torch.Tensor]], torch.Tensor, int, int]:
        with torch.no_grad():
            outputs = model(
                input_ids=current_tokens.unsqueeze(-1),
                past_key_values=past_key_values,
                return_dict=True,
                output_hidden_states=True,
            )
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            hidden_state = outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_size]

            confidence = compute_confidence(
                logits=logits,
                prev_state=prev_hidden_state,
                new_state=hidden_state,
                conf_method=generation_config.conf_method
            )
            exit_now = should_exit(confidence, generation_config.conf_threshold)  # [batch_size]

            if exit_now.any():
                reduced_layers = generation_config.min_exit_layer
                reduced_outputs = model(
                    input_ids=current_tokens.unsqueeze(-1),
                    past_key_values=past_key_values,
                    return_dict=True,
                    output_hidden_states=True,
                    num_layers=reduced_layers
                )
                reduced_logits = reduced_outputs.logits[:, -1, :]
                next_token = torch.argmax(reduced_logits, dim=-1)
                updated_past_key_values = reduced_outputs.past_key_values
                number_of_matches = 1
                draft_num = 1
                output_ids = [next_token.item()]
                new_hidden_state = reduced_outputs.hidden_states[-1][:, -1, :]
            else:
                next_token = torch.argmax(logits, dim=-1)
                updated_past_key_values = outputs.past_key_values
                number_of_matches = 0
                draft_num = 0
                output_ids = [next_token.item()]
                new_hidden_state = hidden_state

            return (
                next_token,
                output_ids,
                updated_past_key_values,
                new_hidden_state,
                number_of_matches,
                draft_num,
            )