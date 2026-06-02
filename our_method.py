import torch
import pandas as pd
import numpy as np
import utils
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import os
import sys
import gc
import math
import random
import logging
from rapidfuzz.distance import Levenshtein as RFLev
from byte_sampling.byte_conditioning import ByteConditioning

from concurrent.futures import ThreadPoolExecutor
from functools import partial

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# bytepredllm_dir = os.path.join(current_dir, 'bytepredllm')
# if bytepredllm_dir not in sys.path:
#     sys.path.insert(0, bytepredllm_dir)

# import of BytePredLLM
from byte_gen_utils import *
from custom_base_model import *
from byte_prediction_model import BytePredLLM
from string_processor.tokenizer_wrapper import ByteTokenizer
from string_processor.string_helper import *
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class MCMCState:
    char_seq: str
    loss: float
    lev_distance: float
    raw_lev_distance: float

class MCMCAttack:

    def __init__(self, model_wrapper, args):
        self.model_wrapper = model_wrapper
        self.args = args
        self.device = args.device
        self.hayse_method = ByteConditioning("meta-llama/Llama-3.2-1B-Instruct")

        self.beta = getattr(args, 'beta', 1.0)
        self.lambda_lev = getattr(args, 'lambda_lev', 0.05)
        self.max_steps = getattr(args, 'max_steps', 50)
        self.burn_in = getattr(args, 'burn_in', 10)
        self.ascii = getattr(args, 'ascii', True)
        
        self.p_ins = getattr(args, 'p_ins', 0.05)
        self.p_del = getattr(args, 'p_del', 0.05)
        self.p_sub = getattr(args, 'p_sub', 0.9)
        self.p_ops = [self.p_ins, self.p_del, self.p_sub]
        self.op_names = ['ins', 'del', 'sub']
        self.lev_max = getattr(args, 'lev_max', None)
        self.base_seed = getattr(args, 'base_seed', 42)
        self.num_chains = 3
        self.early_stop = getattr(args, 'early_stop', True)

        self._T_promptbegin = None
        self._T_promptend = None
        self._T_label = None
        self._criterion = None
        self._orig_label = None
        self._premise = None

    def _setup_loss_context(self, orig_label_int: int, premise: Optional[str], target_class: Optional[int]):
            
            self._orig_label = orig_label_int
            self._premise = premise
            
            llm_proc = utils.llm_input_processor(self.args.dataset, premise, self.args.model_name)
            self._T_promptbegin = self.model_wrapper.tokenizer(
                llm_proc.promptbegin, return_tensors='pt', add_special_tokens=True, truncation=True
            ).to(self.device)
            self._T_promptend = self.model_wrapper.tokenizer(
                llm_proc.promptend, return_tensors='pt', add_special_tokens=False
            ).to(self.device)
            self._T_label = {
                'input_ids': torch.tensor([[orig_label_int]]).to(self.device),
                'attention_mask': torch.tensor([[1]]).to(self.device)
            }
            
            if self.args.loss == 'ce':
                self._criterion = torch.nn.CrossEntropyLoss()
            elif self.args.loss == 'margin' and target_class is not None:
                self._criterion = utils.MarginLossLLM(torch.tensor([target_class]).to(self.device), self.args.tau)
            else:
                self._criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def _compute_loss(self, x_str: str) -> float:
        with torch.no_grad():
            if self._premise is not None and getattr(self.args, 'dataset', '') in ['mnli', 'rte', 'qnli']:
                T_questions = self.model_wrapper.tokenizer(
                    [self._premise], [x_str], return_tensors='pt',
                    padding='longest', add_special_tokens=True, truncation=True
                ).to(self.device)
            else:
                T_questions = self.model_wrapper.tokenizer(
                    [x_str], return_tensors='pt', add_special_tokens=False, truncation=True
                ).to(self.device)

            
            T_all = utils.concat_prompt_question_label(
                self._T_promptbegin, T_questions, self._T_promptend, self._T_label
            )
            outputs = self.model_wrapper.model(
                input_ids=T_all['input_ids'],
                attention_mask=T_all['attention_mask']
            )
            last_logits = outputs.logits[:, -1, :]
            label_tensor = self._T_label['input_ids'].squeeze(-1)
            loss = self._criterion(last_logits, label_tensor)
            
            return loss.mean().item()

    def _char_to_byte(self, char: str) -> int:
        b = char.encode('utf-8')
        assert len(b) == 1
        return b[0]

    def _byte_to_char(self, byte_id: int) -> str:
        return bytes([byte_id]).decode('utf-8')

    def _get_valid_byte_prefix(self, char_seq, char_pos):
        prefix_str = char_seq[:char_pos]
        return list(prefix_str.encode("utf-8"))

    @torch.no_grad()
    def _get_byte_dist(self, prefix):

        sampler = self.hayse_method.get_bytewise_sampler(batch_size=1)
        sampler.add_context([prefix])

        logprobs = sampler.get_dists()[0][:-1]
        assert len(logprobs) == 256, f"Expected 256 bytes, got {len(logprobs)}"
        logprobs = logprobs - torch.logsumexp(logprobs, dim=-1)

        return logprobs.detach().cpu()   
    
    # not ASCII-only functions
    def _sample_char(self, char_seq, pos):
        prefix_bytes = self._get_valid_byte_prefix(char_seq, pos)  
        log_q = 0.0
        byte_seq = []

        for _ in range(4):
            dist = self._get_byte_dist(prefix_bytes + byte_seq)
            b = torch.multinomial(dist, 1).item()
            byte_seq.append(b)
            log_q += torch.log(dist[b].item() + 1e-12).item()
            try:
                char = bytes(byte_seq).decode('utf-8')
                if char.isprintable():
                    return char, log_q, byte_seq
            except UnicodeDecodeError:
                continue
        # если так и не получили валидный символ
        return None, -float('inf'), []
    
    def _compute_char(self, char_seq, pos, target_char):
        prefix_bytes = self._get_valid_byte_prefix(char_seq, pos)  
        log_q = 0.0
        current_bytes = prefix_bytes.copy()

        for b in target_char.encode('utf-8'):
            dist = self._get_byte_dist(current_bytes)
            log_q += torch.log(dist[b].item() + 1e-12).item()
            current_bytes.append(b)
        
        return log_q
    
    def _propose_substitution_all(self, state: MCMCState):
        n = len(state.char_seq)        
        pos = random.randint(0, n - 1)
        old_char = state.char_seq[pos]
        
        new_char, log_q_fwd_sample, _ = self._sample_char(state.char_seq, pos)
            
        new_seq = state.char_seq[:pos] + new_char + state.char_seq[pos+1:]
        
        log_q_bwd_sample = self._compute_char(state.char_seq, pos, old_char)
        
        log_q_fwd = math.log(self.p_ops[2]) - math.log(n) + log_q_fwd_sample
        log_q_bwd = math.log(self.p_ops[2]) - math.log(n) + log_q_bwd_sample
        
        return new_seq, log_q_fwd, log_q_bwd    

    def _propose_insertion_all(self, state: MCMCState):
        n = len(state.char_seq)
        pos = random.randint(0, n)
        
        new_char, log_q_fwd_sample, _ = self._sample_char(state.char_seq, pos)
            
        new_seq = state.char_seq[:pos] + new_char + state.char_seq[pos:]
        
        # q_fwd = P(ins) * (1/(n+1)) * P(new_char)
        log_q_fwd = math.log(self.p_ops[0]) - math.log(n + 1) + log_q_fwd_sample
        # q_bwd = P(del) * (1/len(new_seq))  (обратное удаление выбирает позицию равномерно)
        log_q_bwd = math.log(self.p_ops[1]) - math.log(len(new_seq))
        
        return new_seq, log_q_fwd, log_q_bwd

    def _propose_deletion_all(self, state: MCMCState) -> Tuple[str, float, float]:
        n = len(state.char_seq)
        if n <= 1: return state.char_seq, -float('inf'), -float('inf')
        
        pos = random.randint(0, n - 1)
        old_char = state.char_seq[pos]
        new_seq = state.char_seq[:pos] + state.char_seq[pos+1:]
        
        # q_fwd = P(del) * (1/n)
        log_q_fwd = math.log(self.p_ops[1]) - math.log(n)
        # q_bwd = P(ins) * (1/(len(new_seq))) * P(old_char)  (обратная вставка)
        log_q_bwd_sample = self._compute_char(new_seq, pos, old_char)
        log_q_bwd = math.log(self.p_ops[0]) - math.log(len(new_seq)) + log_q_bwd_sample
        
        return new_seq, log_q_fwd, log_q_bwd


    def _mh_step_all(self, state: MCMCState) -> MCMCState:
        op_idx = np.random.choice(3, p=self.p_ops)
        prop_func = [self._propose_insertion, self._propose_deletion, self._propose_substitution][op_idx]
        
        new_seq, log_q_fwd, log_q_bwd = prop_func(state)
        
        if log_q_fwd == -float('inf') or log_q_bwd == -float('inf') or new_seq == state.char_seq:
            return state

        new_loss = self._compute_loss(new_seq)
        new_lev = RFLev.normalized_similarity(new_seq, self.orig_S)
        raw_lev = RFLev.distance(new_seq, self.orig_S)
        
        if self.lev_max is None or raw_lev <= self.lev_max:
                energy_prop = new_loss + self.lambda_lev * new_lev
                energy_curr = state.loss + self.lambda_lev * state.lev_distance
        
                log_alpha = -self.beta * (energy_prop - energy_curr) + log_q_bwd - log_q_fwd

                if math.log(random.random()) < min(0.0, log_alpha):
                    print(f"[ACCEPT] op={self.op_names[op_idx]}, loss={new_loss:.3f}, lev={new_lev:.3f}, log_alpha={log_alpha:.3f}, seq='{new_seq[:40]}...'")
                    return MCMCState(char_seq=new_seq, loss=new_loss, lev_distance=new_lev)
                else:
                    print(f"[REJECT] op={self.op_names[op_idx]}, log_alpha={log_alpha:.3f}")
                    return state
        else:
            print(f"[REJECT LEV] op={self.op_names[op_idx]}, lev={raw_lev}")
            return state
     

    #-------------------------------------------- 
    # ASCII-only (Log-space implementation)
    #-------------------------------------------- 

    def _mask_dist_log(self, log_probs: torch.Tensor) -> torch.Tensor:

            ascii_mask = torch.zeros_like(log_probs)
            ascii_mask[32:127] = 1.0
            
            masked_log_probs = torch.where(ascii_mask == 1, log_probs, torch.full_like(log_probs, float('-inf')))
            log_norm = torch.logsumexp(masked_log_probs, dim=-1)
            return masked_log_probs - log_norm
    
    def _propose_substitution(self, state: MCMCState) -> Tuple[str, float, float]:
        n = len(state.char_seq)

        pos = random.randint(0, n - 1)
        prefix_bytes = self._get_valid_byte_prefix(state.char_seq, pos)

        log_probs = self._get_byte_dist(prefix_bytes)
        log_ascii_probs = self._mask_dist_log(log_probs)

        new_byte = torch.multinomial(torch.exp(log_ascii_probs), 1).item()
        new_char = self._byte_to_char(new_byte)
        new_seq = state.char_seq[:pos] + new_char + state.char_seq[pos+1:]

        # log q_fwd = log(P_sub) + log(1/n) + log P(new_byte | prefix)
        log_q_fwd = math.log(self.p_ops[2]) - math.log(n) + log_ascii_probs[new_byte].item()
        
        # log q_bwd = log(P_sub) + log(1/n) + log P(old_byte | prefix)
        old_byte = self._char_to_byte(state.char_seq[pos])
        log_q_bwd = math.log(self.p_ops[2]) - math.log(n) + log_ascii_probs[old_byte].item()

        return new_seq, log_q_fwd, log_q_bwd

    def _propose_deletion(self, state: MCMCState) -> Tuple[str, float, float]:
        n = len(state.char_seq)

        pos = random.randint(0, n - 1)
        deleted_char = state.char_seq[pos]
        deleted_byte = self._char_to_byte(deleted_char)
        new_seq = state.char_seq[:pos] + state.char_seq[pos+1:]

        # log q_fwd = log(P_del) + log(1/n)
        log_q_fwd = math.log(self.p_ops[1]) - math.log(n)

        prefix_bytes = self._get_valid_byte_prefix(new_seq, pos)
        log_probs = self._get_byte_dist(prefix_bytes)
        log_ascii_probs = self._mask_dist_log(log_probs)

        # log q_bwd = log(P_ins) + log(1/(len(new))+1) + log P(deleted_byte | prefix)
        log_q_bwd = math.log(self.p_ops[0]) - math.log(len(new_seq)+1) + log_ascii_probs[deleted_byte].item()

        return new_seq, log_q_fwd, log_q_bwd

    def _propose_insertion(self, state: MCMCState) -> Tuple[str, float, float]:
        n = len(state.char_seq)
        pos = random.randint(0, n)

        prefix_bytes = self._get_valid_byte_prefix(state.char_seq, pos)
        log_probs = self._get_byte_dist(prefix_bytes)
        log_ascii_probs = self._mask_dist_log(log_probs)

        new_byte = torch.multinomial(torch.exp(log_ascii_probs), 1).item()
        new_char = self._byte_to_char(new_byte)
        new_seq = state.char_seq[:pos] + new_char + state.char_seq[pos:]

        # log q_fwd = log(P_ins) + log(1/(n+1)) + log P(new_byte | prefix)
        log_q_fwd = math.log(self.p_ops[0]) - math.log(n + 1) + log_ascii_probs[new_byte].item()
        
        # log q_bwd = log(P_del) + log(1/len(new_seq))
        log_q_bwd = math.log(self.p_ops[1]) - math.log(len(new_seq))

        return new_seq, log_q_fwd, log_q_bwd
    
    def _mh_step(self, state: MCMCState) -> MCMCState:
            op_idx = np.random.choice(3, p=self.p_ops)
            prop_funcs = [self._propose_insertion, self._propose_deletion, self._propose_substitution]
            new_seq, log_q_fwd, log_q_bwd = prop_funcs[op_idx](state)

            if (math.isinf(log_q_fwd) and log_q_fwd < 0) or \
            (math.isinf(log_q_bwd) and log_q_bwd < 0) or \
            new_seq == state.char_seq:
                return state

            new_loss = self._compute_loss(new_seq)
            raw_lev = RFLev.distance(new_seq, self.orig_S)
            new_lev = RFLev.normalized_similarity(new_seq, self.orig_S)
            
            if  self.lev_max is None or raw_lev <= self.lev_max:
                energy_prop = new_loss + self.lambda_lev * new_lev
                energy_curr = state.loss + self.lambda_lev * state.lev_distance
                log_alpha = -self.beta * (energy_prop - energy_curr) + log_q_bwd - log_q_fwd
                
                if math.log(random.random()) < min(0.0, log_alpha):
                    print(f"[ACCEPT] op={self.op_names[op_idx]}, loss={new_loss:.3f}, lev={new_lev:.3f}, log_alpha={log_alpha:.3f}, seq='{new_seq[:40]}...'")
                    return MCMCState(char_seq=new_seq, loss=new_loss, lev_distance=new_lev, raw_lev_distance=raw_lev)
                else:
                    print(f"[REJECT] op={self.op_names[op_idx]}, log_alpha={log_alpha:.3f}")
                    return state
            else:
                print(f"[REJECT LEV] op={self.op_names[op_idx]}, lev={raw_lev}")
                return state


# ----- общая часть -----

    def _run_mcmc(self, init_state: MCMCState, seed) -> MCMCState:
        current = init_state
        energy_history = []
        step_func = self._mh_step if self.ascii == True else self._mh_step_all
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
        success = False
        steps = 0

        start_time = time.time()         
        for i in range(self.max_steps):
            steps += 1
            current = step_func(current)
            curr_energy = current.loss + self.lambda_lev * current.lev_distance
            energy_history.append(curr_energy)
            print(f"curr loss: {current.loss}, lev: {current.lev_distance}, orig_lev: {current.raw_lev_distance}")
            if i % 10 == 0:
                print(f"Step {i}, avg energy (last 10): {np.mean(energy_history[-10:]):.3f}")
            if self.early_stop and current.loss < 0.0:
                success = True
                break
        end_time = time.time() - start_time
    
        result = {
            "seed": seed,
            "steps": steps,
            "success": success,
            "loss": current.loss,
            "lev_distance": current.lev_distance,
            "lev_distance_raw": current.raw_lev_distance, 
            "energy": curr_energy,
            "adv_text": current.char_seq,
            "time": end_time
        }
        return result

    def _run_mcmc_with_xpu(self, init_state, seed, chain_idx):
        """Run MCMC chain with XPU support"""
        # Set seeds for this thread
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Handle XPU seeding if available
        if torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
            # Optional: set device for this thread
            # torch.xpu.set_device(chain_idx % torch.xpu.device_count())
        
        current = MCMCState(
            char_seq=init_state.char_seq,
            loss=init_state.loss,
            lev_distance=init_state.lev_distance,
            raw_lev_distance=init_state.raw_lev_distance
        )
        
        energy_history = []
        step_func = self._mh_step if self.ascii == True else self._mh_step_all
        success = False
        steps = 0
        
        start_time = time.time()
        
        for i in range(self.max_steps):
            steps += 1
            current = step_func(current)
            curr_energy = current.loss + self.lambda_lev * current.lev_distance
            energy_history.append(curr_energy)
            print(f"[Chain {chain_idx}] Step {i}: loss={current.loss:.3f}, lev={current.lev_distance:.3f}, orig_lev={current.raw_lev_distance}")
            
            if i % 10 == 0:
                print(f"[Chain {chain_idx}] Step {i}, avg energy (last 10): {np.mean(energy_history[-10:]):.3f}")
            
            if self.early_stop and current.loss < 0.0:
                success = True
                break
        
        end_time = time.time() - start_time
        
        result = {
            "seed": seed,
            "steps": steps,
            "success": success,
            "loss": current.loss,
            "lev_distance": current.lev_distance,
            "lev_distance_raw": current.raw_lev_distance,
            "energy": curr_energy,
            "adv_text": current.char_seq,
            "time": end_time
        }
        return result

    def attack(self, orig_S, orig_label, premise=None, target_class=None, return_us=False):
      
        label_int = orig_label.item() if isinstance(orig_label, torch.Tensor) else int(orig_label)
        self._premise = premise 
        self._setup_loss_context(label_int, premise, target_class)
        self.orig_S = orig_S
        all_iters = []
        init_loss = self._compute_loss(orig_S)
        print(f'initial loss: {init_loss}')
        if init_loss < 0.0:
            return orig_S, 'skip', []
        init_lev = 0.0
        init_orig_lev = 0.0
        init_state = MCMCState(char_seq=orig_S, loss=init_loss, lev_distance=init_lev, raw_lev_distance=init_orig_lev)
        
        best_result = None
        with ThreadPoolExecutor(max_workers=self.num_chains) as executor:
            futures = {}
            for chain_idx in range(self.num_chains):
                current_seed = self.base_seed + chain_idx
                print(f"Submitting Chain {chain_idx + 1}/{self.num_chains} with Seed {current_seed}")
                future = executor.submit(self._run_mcmc_with_xpu, init_state, current_seed, chain_idx)
                futures[future] = chain_idx
        
            for future in futures:
                chain_idx = futures[future]
                try:
                    res = future.result()
                    res["chain_idx"] = chain_idx
                    res["orig_label"] = label_int
                    res["original_text"] = self.orig_S
                    all_iters.append(res)
                    print(f"Chain {chain_idx + 1} completed")
                except Exception as e:
                    print(f"Chain {chain_idx + 1} failed with error: {e}")
                    import traceback
                    traceback.print_exc()
                    
        for res in all_iters:
            if res["success"]:
                if best_result is None or not best_result["success"]:
                    best_result = res
                elif res["steps"] < best_result["steps"]:
                    best_result = res
            else:
                best_result = res
            if best_result is None:
                best_result = res
            elif res["success"] and not best_result["success"]:
                best_result = res
            elif res["success"] == best_result["success"]:
                if res["steps"] < best_result["steps"]:
                    best_result = res 

        adv_S = best_result["adv_text"]

        with torch.no_grad():
            if self._premise is not None and getattr(self.args, 'dataset', '') in ['mnli', 'rte', 'qnli']:
                T_eval = self.model_wrapper.tokenizer(
                    [self._premise], [adv_S], return_tensors='pt',
                    padding='longest', add_special_tokens=True, truncation=True
                ).to(self.device)
            else:
                T_eval = self.model_wrapper.tokenizer(
                    [adv_S], return_tensors='pt', add_special_tokens=False, truncation=True
                ).to(self.device)
                
            T_all_eval = utils.concat_prompt_question_label(
                self._T_promptbegin, T_eval, self._T_promptend, self._T_label
            )
            out = self.model_wrapper.model(
                input_ids=T_all_eval['input_ids'], 
                attention_mask=T_all_eval['attention_mask']
            ).logits
            final_logits = out[0, -1, :]
            adv_label = torch.argmax(final_logits, dim=-1).item()
    
        return adv_S, adv_label, all_iters