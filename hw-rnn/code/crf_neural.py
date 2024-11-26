#!/usr/bin/env python3

# CS465 at Johns Hopkins University.

# Subclass ConditionalRandomFieldBackprop to get a biRNN-CRF model.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf, log, exp
from pathlib import Path
from typing import List, Optional, Tuple
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float

from corpus import IntegerizedSentence, Sentence, Tag, TaggedCorpus, Word
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
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        # [doctring inherited from parent method]

        if unigram:
            raise NotImplementedError("Not required for this homework")

        self.device = device
        self.rnn_dim = rnn_dim
        self.e = lexicon.size(1) if lexicon is not None else len(vocab) - 2
        self.E = lexicon if lexicon is not None else torch.eye(len(vocab) - 2)
    

        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)

        # Create tag embeddings after super().__init__ since it sets self.k
        self.tag_embeddings = torch.eye(self.k)
        
        # Move tensors to GPU if available
        self.E = self.E.to(self.device)
        self.tag_embeddings = self.tag_embeddings.to(self.device)
        
        # Move model to device
        self.to(self.device)

    @override
    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Return the conditional log-probability log p(tags | words)."""
        isent = self._integerize_sentence(sentence, corpus)
        numerator = self.forward_pass(isent)  # Follows gold path if supervised
        
        # For denominator, remove all tag information
        desup_isent = self._integerize_sentence(sentence.desupervise(), corpus)
        denominator = self.forward_pass(desup_isent)  # Marginalizes over all paths

        return numerator - denominator

    @override
    def init_params(self) -> None:

        """
            Initialize all the parameters you will need to support a bi-RNN CRF
            This will require you to create parameters for M, M', U_a, U_b, theta_a
            and theta_b. Use xavier uniform initialization for the matrices and 
            normal initialization for the vectors. 
        """

        # See the "Parameterization" section of the reading handout to determine
        # what dimensions all your parameters will need.

        # rnn matrices for prefix/suffix computation
        self.M = nn.Parameter(torch.randn(self.rnn_dim, 1 + self.rnn_dim + self.e) * 0.1)
        self.M_prime = nn.Parameter(torch.randn(self.rnn_dim, 1 + self.e + self.rnn_dim) * 0.1)

        #one hot tag embeddings
        self.tag_embeddings = torch.eye(self.k).to(self.device)

        # projection mats for feature functions
        feature_dim = 1 + self.rnn_dim + 2*self.k + self.rnn_dim  # for transition features
        self.U_A = nn.Parameter(torch.randn(self.k * self.k, feature_dim) * 0.1)
        
        feature_dim = 1 + self.rnn_dim + self.k + self.e + self.rnn_dim  # for emission features
        self.U_B = nn.Parameter(torch.randn(self.k, feature_dim) * 0.1)


        #emission and transition potentials
        self.theta_A = nn.Parameter(torch.randn(self.k * self.k) * 0.1)
        self.theta_B = nn.Parameter(torch.randn(self.k) * 0.1)
        
    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        # [docstring will be inherited from parent]
    
        # Use AdamW optimizer for better training stability
        self.optimizer = torch.optim.AdamW( 
            params=self.parameters(),       
            lr=lr, 
            weight_decay=weight_decay
        )                                   
        self.scheduler = None            
       
    @override
    def updateAB(self) -> None:
        # Nothing to do - self.A and self.B are not used in non-stationary CRFs
        pass
    
    @override
    def forward_pass(self, isent: IntegerizedSentence) -> TorchScalar:
        self.setup_sentence(isent)
        word_ids = torch.tensor([w for w, _ in isent[1:-1]], 
                          dtype=torch.long, 
                          device='cuda' if self.use_cuda else 'cpu')
        tag_ids = torch.tensor([t if t is not None else -1 for _, t in isent[1:-1]], 
                         dtype=torch.long, 
                         device='cuda' if self.use_cuda else 'cpu')
        T = len(word_ids) + 1

        alpha = torch.full((T, self.k), float('-inf'), device=self.device)
        alpha[0, self.bos_t] = 0.0

        eps = 1e-8  # Numerical stability constant
        is_supervised = any(t is not None for t in tag_ids)

        for t in range(1, T):
            A = self.A_at(t, isent)
            B = self.B_at(t, isent)
            prev_alpha = alpha[t-1]

            if is_supervised and tag_ids[t-1] is not None:
                if t == 1:
                    alpha[t, tag_ids[t-1]] = (prev_alpha[self.bos_t] + 
                                            torch.log(A[self.bos_t, tag_ids[t-1]] + eps) +
                                            torch.log(B[tag_ids[t-1], word_ids[t-1]] + eps))
                else:
                    if tag_ids[t-2] is not None:
                        alpha[t, tag_ids[t-1]] = (prev_alpha[tag_ids[t-2]] + 
                                                torch.log(A[tag_ids[t-2], tag_ids[t-1]] + eps) +
                                                torch.log(B[tag_ids[t-1], word_ids[t-1]] + eps))
            else:
                alpha_t = torch.logsumexp(prev_alpha.unsqueeze(1) + torch.log(A + eps), dim=0)
                alpha[t] = alpha_t + torch.log(B[:, word_ids[t-1]] + eps)

        final_A = self.A_at(T, isent)
        if is_supervised and tag_ids[-1] is not None:
            self.log_Z = alpha[T-1, tag_ids[-1]] + torch.log(final_A[tag_ids[-1], self.eos_t] + eps)
        else:
            self.log_Z = torch.logsumexp(alpha[T-1] + torch.log(final_A[:, self.eos_t] + eps), dim=0)

        self.alpha = alpha
        return self.log_Z
    
    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Pre-compute the biRNN prefix and suffix contextual features (h and h'
        vectors) at all positions, as defined in the "Parameterization" section
        of the reading handout.  They can then be accessed by A_at() and B_at().
        
        Make sure to call this method from the forward_pass, backward_pass, and
        Viterbi_tagging methods of HiddenMarkovMOdel, so that A_at() and B_at()
        will have correct precomputed values to look at!"""

        word_ids = torch.tensor([w for w, _ in isent])
        word_embeddings = self.E[word_ids]
        T = len(word_ids)
        
        #forward pass 
        h_forward = torch.zeros(T, self.rnn_dim, device=self.device)
        h_curr = torch.zeros(self.rnn_dim, device=self.device)
        
        for j in range(T):
            input_vec = torch.cat([
                torch.ones(1, device=self.device),
                h_curr,
                word_embeddings[j]
            ])
            h_curr = torch.sigmoid(self.M @ input_vec)
            h_forward[j] = h_curr

        # Backward pass (vectorized)
        h_backward = torch.zeros(T, self.rnn_dim, device=self.device)
        h_curr = torch.zeros(self.rnn_dim, device=self.device)
        
        for j in range(T-1, -1, -1):
            input_vec = torch.cat([
                torch.ones(1, device=self.device),
                word_embeddings[j],
                h_curr
            ])
            h_curr = torch.sigmoid(self.M_prime @ input_vec)
            h_backward[j] = h_curr

        self.h_forward = h_forward
        self.h_backward = h_backward
        self.curr_sent = isent

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        """Computes non-stationary k x k transition potential matrix using biRNN 
        contextual features and tag embeddings (one-hot encodings). Output should 
        be ϕA from the "Parameterization" section in the reading handout."""

        if not hasattr(self, 'curr_sent') or self.curr_sent != sentence:
            self.setup_sentence(sentence)

        # Get contextual features
        h_prefix = self.h_forward[position-1] if position > 0 else torch.zeros(self.rnn_dim, device=self.device)
        h_suffix = self.h_backward[position] if position < len(sentence) else torch.zeros(self.rnn_dim, device=self.device)

        # Create tag combination indices (vectorized)
        s_indices = torch.arange(self.k, device=self.device).unsqueeze(1).repeat(1, self.k).reshape(-1)
        t_indices = torch.arange(self.k, device=self.device).repeat(self.k)

        # Create one-hot encodings (vectorized)
        s_emb = F.one_hot(s_indices, num_classes=self.k).float().to(self.device)
        t_emb = F.one_hot(t_indices, num_classes=self.k).float().to(self.device)

        # Expand contextual features
        h_prefix_expanded = h_prefix.unsqueeze(0).expand(self.k * self.k, -1)
        h_suffix_expanded = h_suffix.unsqueeze(0).expand(self.k * self.k, -1)

        # Compute feature matrix (vectorized)
        feature_matrix = torch.cat([
            torch.ones(self.k * self.k, 1, device=self.device),
            h_prefix_expanded,
            s_emb,
            t_emb,
            h_suffix_expanded
        ], dim=1)

        # Compute potentials (vectorized)
        f_A = torch.sigmoid(self.U_A @ feature_matrix.T).sum(dim=0)
        potentials = torch.exp(self.theta_A * f_A).reshape(self.k, self.k)
        potentials = torch.clamp(potentials, min=1e-6, max=1e6)
        # Apply mask for structural zeros
        mask = torch.ones_like(potentials, device=self.device)
        mask[:, self.bos_t] = 0
        mask[self.eos_t, :] = 0

        
        return potentials * mask

    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        """Compute emission potentials for position j."""
        if position >= len(sentence):
            return torch.zeros(self.k, self.V)
            
        if not hasattr(self, 'curr_sent') or self.curr_sent != sentence:
            self.setup_sentence(sentence)

        word_id = sentence[position][0]
        word_emb = self.E[word_id]
        
        #contextual features
        h_prefix = self.h_forward[position-1] if position > 0 else torch.zeros(self.rnn_dim, device=self.device)
        h_suffix = self.h_backward[position] if position < len(sentence) else torch.zeros(self.rnn_dim, device=self.device)

        # Create feature matrix (vectorized)
        feature_matrix = torch.cat([
            torch.ones(self.k, 1, device=self.device),
            h_prefix.unsqueeze(0).expand(self.k, -1),
            self.tag_embeddings,
            word_emb.unsqueeze(0).expand(self.k, -1),
            h_suffix.unsqueeze(0).expand(self.k, -1)
        ], dim=1)

        # Compute potentials (vectorized)
        f_B = torch.sigmoid(self.U_B @ feature_matrix.T).sum(dim=0)
        potentials = torch.exp(self.theta_B * f_B).unsqueeze(1).expand(-1, self.V)

        # Apply mask for structural zeros
        mask = torch.ones_like(potentials, device=self.device)
        mask[self.eos_t:, :] = 0

        return potentials * mask
