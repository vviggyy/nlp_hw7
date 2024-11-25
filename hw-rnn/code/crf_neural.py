#!/usr/bin/env python3

# CS465 at Johns Hopkins University.

# Subclass ConditionalRandomFieldBackprop to get a biRNN-CRF model.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf, log, exp
from pathlib import Path
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
                 unigram: bool = False):
        # [doctring inherited from parent method]

        if unigram:
            raise NotImplementedError("Not required for this homework")

        self.rnn_dim = rnn_dim
        self.e = lexicon.size(1) if lexicon is not None else len(vocab) - 2  # one-hot dimension if no lexicon
        self.E = lexicon if lexicon is not None else torch.eye(len(vocab) - 2)  # use one-hot if no lexicon
        

        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)

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
        self.tag_embeddings = torch.eye(self.k)

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
            lr=lr, weight_decay=weight_decay
        )                                   
        self.scheduler = None            
       
    @override
    def updateAB(self) -> None:
        # Nothing to do - self.A and self.B are not used in non-stationary CRFs
        pass
    
    @override
    def forward_pass(self, isent: IntegerizedSentence) -> TorchScalar:
        self.setup_sentence(isent)
        word_ids = torch.tensor([w for w, _ in isent[1:-1]], dtype=torch.long)
        tag_ids = [t for _, t in isent[1:-1]]  # Keep as None if unsupervised
        T = len(word_ids) + 1
        
        alpha = torch.full((T, self.k), float('-inf'))
        alpha[0, self.bos_t] = 0.0
        
        # Check if this is a supervised sequence
        is_supervised = any(t is not None for t in tag_ids)
        
        for t in range(1, T):
            A = self.A_at(t, isent)
            B = self.B_at(t, isent)
            prev_alpha = alpha[t-1]
            
            if is_supervised and tag_ids[t-1] is not None:
                # For supervised steps, directly accumulate the score
                if t == 1:
                    # From BOS to first tag
                    alpha[t, tag_ids[t-1]] = (alpha[0, self.bos_t] + 
                                            torch.log(A[self.bos_t, tag_ids[t-1]] + 1e-10) +
                                            torch.log(B[tag_ids[t-1], word_ids[t-1]] + 1e-10))
                else:
                    # From previous gold tag to current gold tag
                    if tag_ids[t-2] is not None:
                        alpha[t, tag_ids[t-1]] = (alpha[t-1, tag_ids[t-2]] + 
                                                torch.log(A[tag_ids[t-2], tag_ids[t-1]] + 1e-10) +
                                                torch.log(B[tag_ids[t-1], word_ids[t-1]] + 1e-10))
            else:
                # For unsupervised steps, consider all possible transitions
                alpha_t = torch.logsumexp(prev_alpha.unsqueeze(1) + torch.log(A + 1e-10), dim=0)
                alpha[t] = alpha_t + torch.log(B[:, word_ids[t-1]] + 1e-10)

        # Handle final transition to EOS
        final_A = self.A_at(T, isent)
        if is_supervised and tag_ids[-1] is not None:
            # Direct transition from final gold tag to EOS
            self.log_Z = alpha[T-1, tag_ids[-1]] + torch.log(final_A[tag_ids[-1], self.eos_t] + 1e-10)
        else:
            # Consider all possible transitions to EOS
            self.log_Z = torch.logsumexp(alpha[T-1] + torch.log(final_A[:, self.eos_t] + 1e-10), dim=0)
        
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
        
        # Forward pass 
        h_forward = torch.zeros(T, self.rnn_dim)
        h_curr = torch.zeros(self.rnn_dim)
        for j in range(T):
            input_vec = torch.cat([torch.ones(1), h_curr, word_embeddings[j]])
            h_curr = torch.sigmoid(self.M @ input_vec)
            h_forward[j] = h_curr

        # Backward pass
        h_backward = torch.zeros(T, self.rnn_dim) 
        h_curr = torch.zeros(self.rnn_dim)
        for j in range(T-1, -1, -1):
            input_vec = torch.cat([torch.ones(1), word_embeddings[j], h_curr])
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

        #get contextual features
        h_prefix = self.h_forward[position-1] if position > 0 else torch.zeros(self.rnn_dim)
        h_suffix = self.h_backward[position] if position < len(sentence) else torch.zeros(self.rnn_dim)
        
        # One-hot tag vectors
        potentials = torch.zeros(self.k, self.k)
        for s in range(self.k):
            s_emb = torch.eye(self.k)[s]
            for t in range(self.k):
                t_emb = torch.eye(self.k)[t]
                
                # Concatenate features as in equation (47)
                f_A = torch.sigmoid(self.U_A @ torch.cat([
                    torch.ones(1),
                    h_prefix,
                    s_emb,
                    t_emb,
                    h_suffix
                ]))
                
                # Compute potential as in equation (45)
                potentials[s,t] = torch.exp(self.theta_A[s*self.k + t] * f_A.sum())
        
        # Set structural zeros
        potentials[:, self.bos_t] = 0
        potentials[self.eos_t, :] = 0

        return potentials
        
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
        
        # Get contextual features
        h_prefix = self.h_forward[position-1] if position > 0 else torch.zeros(self.rnn_dim)
        h_suffix = self.h_backward[position] if position < len(sentence) else torch.zeros(self.rnn_dim)

        potentials = torch.zeros(self.k, self.V)
        for t in range(self.k):
            t_emb = torch.eye(self.k)[t]
            
            # Concatenate features as in equation (48)
            f_B = torch.sigmoid(self.U_B @ torch.cat([
                torch.ones(1),
                h_prefix,
                t_emb,
                word_emb,
                h_suffix
            ]))
            
            # Compute potential as in equation (45)
            potential = torch.exp(self.theta_B[t] * f_B.sum())
            potentials[t] = potential.repeat(self.V)

        # Set structural zeros
        potentials[self.eos_t:, :] = 0
        
        return potentials
