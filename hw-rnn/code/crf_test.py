#!/usr/bin/env python3

# Subclass ConditionalRandomFieldBackprop to get a model that uses some
# contextual features of your choice.  This lets you test the revision to hmm.py
# that uses those features.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import tensor, Tensor, cuda
from jaxtyping import Float

from corpus import Tag, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldTest(ConditionalRandomFieldBackprop):
    """A CRF with some arbitrary non-stationary features, for testing."""
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        """Construct an CRF with initially random parameters, with the
        given tagset, vocabulary, and lexical features.  See the super()
        method for discussion."""
        if unigram:
            raise NotImplementedError("Not required for this homework")
            

        self.E = lexicon          # rows are word embeddings
        self.e = lexicon.size(1)  # dimensionality of word embeddings
        self.rnn_dim = rnn_dim        
        torch.autograd.set_detect_anomaly(True)
        # an __init__() call to the nn.Module class must be made before assignment on the child.
        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)

    @override
    def init_params(self) -> None:
        # [docstring will be inherited from parent method]
        self.WA = nn.Parameter(torch.randn(self.k, self.k) * 0.1)
        self.WB = nn.Parameter(torch.randn(self.k, self.V) * 0.1)
        
        # Position-aware features
        self.pos_dim = 4
        self.pos_embeddings = nn.Parameter(torch.randn(50, self.pos_dim) * 0.1)
        self.trans_proj = nn.Parameter(torch.randn(self.k * self.k, self.e + self.pos_dim) * 0.1)
        self.emit_proj = nn.Parameter(torch.randn(self.k, self.e + self.pos_dim) * 0.1)
        
        self.updateAB()


    @override
    def updateAB(self) -> None:
        # Your non-stationary A_at() and B_at() might not make any use of the
        # stationary A and B matrices computed by the parent.  So we override
        # the parent so that we won't waste time computing self.A, self.B.
        #
        # But if you decide that you want A_at() and B() at to refer to self.A
        # and self.B (for example, multiplying stationary and non-stationary
        # potentials), then you'll still need to compute them; in that case,
        # don't override the parent in this way.
        A = torch.exp(self.WA).clone()
        B = torch.exp(self.WB).clone()
        
        # Apply structural zeros to copies
        A[:, self.bos_t] = 0
        A[self.eos_t, :] = 0
        B[self.eos_t:, :] = 0
        
        self.A = A
        self.B = B

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        """Compute transition potentials at given position."""
        
        curr_word = sentence[position][0]
        curr_emb = self.E[curr_word]
        
        pos = min(position, len(self.pos_embeddings)-1)
        pos_emb = self.pos_embeddings[pos]
        
        combined = torch.cat([curr_emb, pos_emb])
        trans_logits = F.linear(combined, self.trans_proj)
        trans_logits = trans_logits.reshape(self.k, self.k)
        
        A = torch.exp(trans_logits)
        A[:, self.bos_t] = 0
        A[self.eos_t, :] = 0
        
        return A

    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        """Compute emission potentials at given position."""
        
        if position >= len(sentence):
            return torch.zeros(self.k, self.V)
            
        word = sentence[position][0]
        word_emb = self.E[word]
        pos = min(position, len(self.pos_embeddings)-1)
        pos_emb = self.pos_embeddings[pos]
        
        combined = torch.cat([word_emb, pos_emb])
        emit_logits = F.linear(combined, self.emit_proj)
        B = torch.exp(emit_logits).unsqueeze(1).expand(-1, self.V)
        B[self.eos_t:, :] = 0
        return B
