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
        self.e = lexicon.size(1) # dimensionality of word's embeddings
        self.E = lexicon

        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)


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

        if self.rnn_dim == 0:
            logger.info("No RNN dimension specified, using lexical features only")
            self.tag_embeddings = torch.eye(self.k)

            # Corrected dimensions for self.UA and self.UB
            self.UA = nn.Parameter(torch.empty(2 * self.k, 1))
            nn.init.xavier_uniform_(self.UA)

            self.UB = nn.Parameter(torch.empty(self.k + self.e, 1))
            nn.init.xavier_uniform_(self.UB)

            self.M = self.Mprime = None
        else:
            # RNN case - implement this later
            raise NotImplementedError("RNN case not implemented yet")

        logger.info(f"Neural CRF dimensions:")
        logger.info(f"Word embeddings: {self.e}, Tags: {self.k}")
        logger.info(f"UA: {self.UA.shape} (maps to {self.k}x{self.k} transitions)")
        logger.info(f"UB: {self.UB.shape} (maps to tag scores)")

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
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Pre-compute the biRNN prefix and suffix contextual features (h and h'
        vectors) at all positions, as defined in the "Parameterization" section
        of the reading handout.  They can then be accessed by A_at() and B_at().
        
        Make sure to call this method from the forward_pass, backward_pass, and
        Viterbi_tagging methods of HiddenMarkovMOdel, so that A_at() and B_at()
        will have correct precomputed values to look at!"""

        # Extract word IDs excluding BOS/EOS
        word_ids = torch.tensor([w for w, _ in isent[1:-1]], dtype=torch.long)
        n = len(word_ids)
        
        if self.rnn_dim > 0:
            # Use RNN to get contextual representations
            self.h_fwd = torch.zeros(n + 1, self.rnn_dim)
            self.h_back = torch.zeros(n + 1, self.rnn_dim)
            
            # Forward pass
            h_prev = torch.zeros(self.rnn_dim)
            for j in range(n):
                word_embed = self.E[word_ids[j]]
                inp = torch.cat([torch.ones(1), h_prev, word_embed])
                self.h_fwd[j] = torch.sigmoid(inp @ self.M)
                h_prev = self.h_fwd[j]
            
            # Backward pass
            h_prev = torch.zeros(self.rnn_dim)
            for j in range(n-1, -1, -1):
                word_embed = self.E[word_ids[j]]
                inp = torch.cat([torch.ones(1), h_prev, word_embed])
                self.h_back[j] = torch.sigmoid(inp @ self.Mprime)
                h_prev = self.h_back[j]
        else:
            # No RNN - just store word embeddings directly
            self.h_fwd = None
            self.h_back = None
        
        # Store sentence info
        self.sent_len = n
        self.word_ids = word_ids

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position: int, sentence: IntegerizedSentence) -> Tensor:
        
        """Computes non-stationary k x k transition potential matrix using biRNN 
        contextual features and tag embeddings (one-hot encodings). Output should 
        be ϕA from the "Parameterization" section in the reading handout."""

        if position == 0:
            # Only transitions from BOS allowed
            trans = torch.full((self.k, self.k), float('-inf'))
            trans[self.bos_t, :] = 0
            trans[self.bos_t, self.eos_t] = float('-inf')  # except to EOS
            return torch.exp(trans)
        elif position > self.sent_len:
            # Only transitions to EOS allowed
            trans = torch.full((self.k, self.k), float('-inf'))
            trans[:, self.eos_t] = 0
            trans[self.eos_t, :] = float('-inf')  # except from EOS
            return torch.exp(trans)

        # Concatenate tag embeddings for all pairs
        row_tags = self.tag_embeddings.unsqueeze(1).expand(-1, self.k, -1)
        col_tags = self.tag_embeddings.unsqueeze(0).expand(self.k, -1, -1)
        tag_pairs = torch.cat([row_tags, col_tags], dim=2)
        
        # Reshape to (k*k, 2k) matrix
        tag_pairs = tag_pairs.reshape(self.k * self.k, 2 * self.k)
        
        # Compute potentials and reshape to k x k
        potentials = torch.sigmoid(tag_pairs @ self.UA).reshape(self.k, self.k)
        
        mask = torch.ones_like(potentials)
        mask[:, self.bos_t] = 0  # Can't transition to BOS
        mask[self.eos_t, :] = 0  # Can't transition from EOS
        mask[self.bos_t, self.eos_t] = 0  # BOS can't go to EOS

        # Element-wise multiplication to apply the mask
        potentials = potentials * mask
        return potentials
        
    @override
    @typechecked
    def B_at(self, position: int, sentence: IntegerizedSentence) -> Tensor:
        """Computes non-stationary k x V emission potential matrix using biRNN 
        contextual features, tag embeddings (one-hot encodings), and word embeddings. 
        Output should be ϕB from the "Parameterization" section in the reading handout."""

        if position == 0 or position > self.sent_len:
            emiss = torch.zeros(self.k, self.V)
            return emiss  # BOS/EOS have no emissions in vocabulary
        
        # Get word embedding for current position
        word_j = self.word_ids[position-1]
        word_embed = self.E[word_j]
        
        # For each tag, concatenate [t; w] and compute score
        features = torch.cat([self.tag_embeddings, 
                            word_embed.unsqueeze(0).expand(self.k, -1)], 
                            dim=1)
        
        # Get scores for each tag
        tag_scores = torch.sigmoid(features @ self.UB).squeeze()

        # Create full emission matrix with scores only at observed word
        B = torch.zeros(self.k, self.V)
        B[:, word_j] = tag_scores
        
        # Apply structural zeros
        B[self.bos_t, :] = 0
        B[self.eos_t, :] = 0
        
        return B
    
    @override 
    def forward_pass(self, isent: IntegerizedSentence) -> TorchScalar:
        """Compute forward probabilities using position-specific potentials."""
        self.setup_sentence(isent)  # Compute RNN states
        
        # Extract word IDs and tags
        word_ids = torch.tensor([w for w, _ in isent[1:-1]], dtype=torch.long)
        tag_ids = [t for _, t in isent[1:-1]]
        T = len(word_ids) + 1

        # Initialize alpha[0] with transitions from BOS
        alpha = torch.zeros(T, self.k)
        alpha[0, self.bos_t] = 0.0  # log(1) = 0
        alpha[0, :self.bos_t] = float('-inf')  # Can't start with other tags
        alpha[0, self.bos_t+1:] = float('-inf')

        # Forward pass using log probabilities
        for t in range(1, T):
            # Get position-specific potentials
            A = self.A_at(t, isent)  # k x k transition matrix
            B = self.B_at(t, isent)  # k x V emission matrix
            
            # Current word ID for emissions
            w = word_ids[t-1]
            
            # Handle supervised case if tag is known
            if tag_ids[t-1] is not None:
                next_tag = tag_ids[t-1]
                # Only allow transition to known tag
                mask = torch.zeros_like(A)
                mask[:, next_tag] = 1
                A = torch.where(mask.bool(), A, torch.tensor(float('-inf')))
            
            # Forward update in log space
            # alpha[t] = logsumexp(alpha[t-1] + log(A)) + log(B[:, w])
            trans_scores = alpha[t-1].unsqueeze(1) + torch.log(A + 1e-12)
            alpha[t] = torch.logsumexp(trans_scores, dim=0) + torch.log(B[:, w] + 1e-12)
            
            # Zero out impossible tags
            alpha[t, self.bos_t] = float('-inf')  # Can't use BOS tag
        
        # Final transition to EOS
        final_A = self.A_at(T, isent)
        if tag_ids[-1] is not None:
            # Only allow known final tag
            final_trans = torch.full_like(final_A[:, self.eos_t], float('-inf'))
            final_trans[tag_ids[-1]] = torch.log(final_A[tag_ids[-1], self.eos_t] + 1e-12)
        else:
            final_trans = torch.log(final_A[:, self.eos_t] + 1e-12)
            
        # Compute final log probability
        self.log_Z = torch.logsumexp(alpha[T-1] + final_trans, dim=0)
        self.alpha = alpha
        
        return self.log_Z
    
