#!/usr/bin/env python3

# CS465 at Johns Hopkins University.

# Subclass ConditionalRandomFieldBackprop to get a biRNN-CRF model.

from __future__ import annotations
from collections import defaultdict
import logging
from typing import List, Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
from math import inf, log, exp
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
torch.autograd.set_detect_anomaly(True)
from torch import Tensor, cuda
from jaxtyping import Float

from corpus import BOS_WORD, EOS_WORD, OOV_WORD, IntegerizedSentence, Sentence, Tag, TaggedCorpus, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldNeural(ConditionalRandomFieldBackprop):
    """A CRF that uses a biRNN to compute non-stationary potential
    matrices.  The feature functions used to compute the potentials
    are now non-stationary, non-linear functions of the biRNN
    parameters."""
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Optional[Tensor] = None,
                 rnn_dim: Optional[int] = None,
                 unigram: bool = False, 
                 corpus: Optional[TaggedCorpus] = None):
        # [doctring inherited from parent method]

        if unigram:
            raise NotImplementedError("Not required for this homework")


        nn.Module.__init__(self)  
        
        self.lexicon = lexicon  # Store temporarily
        self.e = lexicon.size(1) 
        self.rnn_dim = rnn_dim

        super().__init__(tagset, vocab, unigram)
        if corpus is not None:
            self.compute_frequency_features(corpus)
    
        

    @override
    def init_params(self) -> None:

        """
            Initialize all the parameters you will need to support a bi-RNN CRF
            This will require you to create parameters for M, M', U_a, U_b, theta_a
            and theta_b. Use xavier uniform initialization for the matrices and 
            normal initialization for the vectors. 
        """
        self.E = nn.Parameter(self.lexicon.to(self.device), requires_grad=False)
        del self.lexicon
        
        # one-hot tag embeddings
        self.tag_embeddings = nn.Parameter(torch.eye(self.k, device=self.device), requires_grad=False)
        
        # rNN parameters
        rnn_input_dim = self.e + self.rnn_dim
        self.M = nn.Parameter(torch.empty(self.rnn_dim, rnn_input_dim, device=self.device))
        self.Mprime = nn.Parameter(torch.empty(self.rnn_dim, rnn_input_dim, device=self.device))
        self.b = nn.Parameter(torch.zeros(self.rnn_dim, device=self.device))
        self.bprime = nn.Parameter(torch.zeros(self.rnn_dim, device=self.device))
        
        nn.init.xavier_uniform_(self.M)
        nn.init.xavier_uniform_(self.Mprime)
        
        # transition and emission parameters
        trans_dim = 2*self.rnn_dim + 2*self.k
        emit_dim = 2*self.rnn_dim + self.k + self.e + self.k + 1  # Added k+1 for frequencies

        self.UA = nn.Parameter(torch.empty(trans_dim, 1, device=self.device))
        self.UB = nn.Parameter(torch.empty(emit_dim, 1, device=self.device))
        
        
        nn.init.xavier_uniform_(self.UA)
        nn.init.xavier_uniform_(self.UB)
        
        self.reset_cached_states()

        logger.info(f"Initialized Neural CRF on device: {self.device}")
        logger.info(f"RNN dim: {self.rnn_dim}, Word embed dim: {self.e}")
        logger.info(f"UA: {self.UA.size()}")
        logger.info(f"UB: {self.UB.size()}")


    def reset_cached_states(self):
        """Reset cached RNN states."""
        self.h_fwd = None
        self.h_back = None
        self.sent_len = None
        self.word_ids = None

    @staticmethod
    @torch.jit.script  # JIT compile for speed
    def rnn_step(x: Tensor, h_prev: Tensor, M: Tensor, b: Tensor) -> Tensor:
        """Single RNN step with optimized computation."""
        return torch.sigmoid(M @ torch.cat([h_prev, x]) + b)
    

    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True
        )
        
        # i dont htink this works well on kaggle.. might comment out 
        self.scaler = torch.cuda.amp.GradScaler()
        
        #  gradient clipping
        self.grad_clip_value = 1.0        
       
    @override
    def updateAB(self) -> None:
        # Nothing to do - self.A and self.B are not used in non-stationary CRFs
        pass
    
    @override
    def forward_pass(self, sentence: IntegerizedSentence) -> Tensor:
        self.setup_sentence(sentence)
        n = len(sentence)
        alpha = torch.full((n, self.k), -inf, device=self.device)
        alpha[0, self.bos_t] = 0

        A_matrices = [self.A_at(j-1, sentence) for j in range(1, n)]
        B_matrices = [self.B_at(j, sentence) for j in range(1, n)]
        
        word_ids = torch.tensor([min(sentence[j][0], self.V-1) for j in range(1, n)], device=self.device)
        
        for j in range(1, n):
            A = A_matrices[j-1]
            B = B_matrices[j-1]
            word_id = word_ids[j-1]
            tag_id = sentence[j][1]

            if tag_id is not None:
                alpha_prev = alpha[j-1]
                alpha_jt = alpha_prev + torch.log(A[:, tag_id] + 1e-10) + torch.log(B[:, word_id] + 1e-10)
                alpha_j = torch.full_like(alpha_prev, -inf)
                alpha_j[tag_id] = torch.logsumexp(alpha_jt, dim=0)
                alpha[j] = alpha_j
            else:
                alpha[j] = torch.logsumexp(
                    alpha[j-1].unsqueeze(1) + 
                    torch.log(A + 1e-10) + 
                    torch.log(B[:, word_id].unsqueeze(0) + 1e-10),
                    dim=0
                )

        return alpha[-1, self.eos_t]
    
    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        word_ids = torch.tensor([w for w, _ in isent], dtype=torch.long, device=self.device)
        self.sent_len = len(word_ids)
        self.word_ids = word_ids
        word_embeds = self.E[word_ids]  # [seq_len, embed_dim]

        # preallo tensors
        self.h_fwd = torch.zeros(self.sent_len, self.rnn_dim, device=self.device)
        self.h_back = torch.zeros(self.sent_len, self.rnn_dim, device=self.device)
        
        #  vectorization
        h = torch.zeros(self.rnn_dim, device=self.device)
        for t in range(self.sent_len):
            h = self.rnn_step(word_embeds[t], h, self.M, self.b)
            self.h_fwd[t] = h

        # back pass
        h = torch.zeros(self.rnn_dim, device=self.device)
        for t in range(self.sent_len-1, -1, -1):
            h = self.rnn_step(word_embeds[t], h, self.Mprime, self.bprime)
            self.h_back[t] = h
    
    #i think i really dont need this - might get rid of entirely 
    def compute_frequency_features(self, corpus: TaggedCorpus) -> None:
        word_counts = defaultdict(int)
        tag_given_word = defaultdict(lambda: defaultdict(int))
        
        # Skip BOS/EOS when counting
        for sentence in corpus:
            for word, tag in sentence:
                if word not in [BOS_WORD, EOS_WORD]:
                    word_counts[word] += 1
                    if tag:
                        tag_given_word[word][tag] += 1
        
        # Initialize tensors without special tokens (V-2)
        self.word_freqs = torch.zeros(self.V-2, device=self.device)
        self.tag_freqs = torch.zeros(self.V-2, self.k, device=self.device)
        
        for word, count in word_counts.items():
            if word in self.vocab:
                idx = self.vocab.index(word)
                if idx >= 2:  # Skip BOS/EOS positions
                    self.word_freqs[idx-2] = count
                    for tag, tag_count in tag_given_word[word].items():
                        tag_idx = self.tagset.index(tag)
                        self.tag_freqs[idx-2, tag_idx] = tag_count
                        
        self.word_freqs = torch.log1p(self.word_freqs)
        self.tag_freqs = torch.log1p(self.tag_freqs)
    
    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    def A_at(self, position: int, sentence: IntegerizedSentence) -> Tensor:
        if position == 0:
            A = torch.zeros(self.k, self.k, device=self.device).clone()
            A[self.bos_t, :] = 1
            A[self.bos_t, self.eos_t] = 0
            return A.clone()
                
        if position > self.sent_len:
            A = torch.zeros(self.k, self.k, device=self.device).clone()
            A[:, self.eos_t] = 1
            A[self.eos_t, :] = 0
            return A.clone()

        h_j = self.h_fwd[position-1]
        h_prime_j = self.h_back[position-1]

        # batchfeatures for all k*k tag pairs
        features = torch.cat([
            h_j.repeat(self.k * self.k, 1),
            h_prime_j.repeat(self.k * self.k, 1),
            self.tag_embeddings.repeat_interleave(self.k, dim=0),
            self.tag_embeddings.repeat(self.k, 1)
        ], dim=1)
        
        scores = (features @ self.UA).squeeze(-1)
        A = torch.softmax(scores.view(self.k, self.k), dim=1).clone()
        
        mask = torch.ones_like(A)
        mask[:, self.bos_t] = 0
        mask[self.eos_t, :] = 0
        mask[self.bos_t, self.eos_t] = 0
        
        return (A * mask).clone()
    
    @override  
    def B_at(self, position: int, sentence: IntegerizedSentence) -> Tensor:
        if position == 0 or position > self.sent_len:
            return torch.zeros(self.k, self.V, device=self.device).clone()
            
        word_id = min(sentence[position][0], self.V-1)
        h_j = self.h_fwd[position-1]
        h_prime_j = self.h_back[position-1]
                
        if word_id >= 2:
            word_freq = self.word_freqs[word_id-2].unsqueeze(0)
            tag_freq = self.tag_freqs[word_id-2]
        else:
            word_freq = torch.zeros(1, device=self.device)
            tag_freq = torch.zeros(self.k, device=self.device)

        features = torch.cat([
            h_j.repeat(self.k, 1),
            h_prime_j.repeat(self.k, 1),
            self.tag_embeddings,
            self.E[word_id].repeat(self.k, 1),
            tag_freq.repeat(self.k, 1),
            word_freq.repeat(self.k, 1)
        ], dim=1)
                
        emission_scores = (features @ self.UB).squeeze(-1)
        B = torch.zeros(self.k, self.V, device=self.device).clone()
        
        mask = torch.ones(self.k, device=self.device)
        mask[self.bos_t] = 0
        mask[self.eos_t] = 0
        
        B[:, word_id] = (torch.softmax(emission_scores, dim=0) * mask).clone()
        return B.clone()