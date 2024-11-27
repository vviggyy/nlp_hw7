#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Starter code for Hidden Markov Models.

from __future__ import annotations
from collections import defaultdict
import logging
from math import inf, log, exp
from pathlib import Path
import os, time
from typing import Callable, List, Optional, cast
from typeguard import typechecked

import torch
from torch import Tensor, cuda, nn
from jaxtyping import Float

from tqdm import tqdm # type: ignore
import pickle

from integerize import Integerizer
from corpus import BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag, TaggedCorpus, IntegerizedSentence, Word

TorchScalar = Float[Tensor, ""] # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

###
# HMM tagger
###
class HiddenMarkovModel:
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the HMM states as "tags" and the HMM observations 
    as "words."
    """
    
    # As usual in Python, attributes and methods starting with _ are intended as private;
    # in this case, they might go away if you changed the parametrization of the model.

    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        """Construct an HMM with initially random parameters, with the
        given tagset, vocabulary, and lexical features.
        
        Normally this is an ordinary first-order (bigram) HMM.  The `unigram` flag
        says to fall back to a zeroth-order HMM, in which the different
        positions are generated independently.  (The code could be extended to
        support higher-order HMMs: trigram HMMs used to be popular.)"""

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if vocab[-2:] != [EOS_WORD, BOS_WORD]:
            raise ValueError("final two types of vocab should be EOS_WORD, BOS_WORD")

        self.k = len(tagset)       # number of tag types
        self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        self.unigram = unigram     # do we fall back to a unigram model?

        self.tagset = tagset
        self.vocab = vocab

        # Useful constants that are referenced by the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        if self.bos_t is None or self.eos_t is None:
            raise ValueError("tagset should contain both BOS_TAG and EOS_TAG")
        assert self.eos_t is not None    # we need this to exist
        self.eye: Tensor = torch.eye(self.k)  # identity matrix, used as a collection of one-hot tag vectors

        self.init_params()     # create and initialize model parameters

    @typechecked
    def A_at(self, position: int, sentence: IntegerizedSentence) -> Tensor:
        """Return transition matrix for a specific position in a sentence."""
        return self.A

    @typechecked
    def B_at(self, position: int, sentence: IntegerizedSentence) -> Tensor:
        """Return emission matrix for a specific position in a sentence."""
        return self.B
 
    def init_params(self) -> None:
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).  
        We respect structural zeroes ("Don't guess when you know").
            
        If you prefer, you may change the class to represent the parameters in logspace,
        as discussed in the reading handout as one option for avoiding underflow; then name
        the matrices lA, lB instead of A, B, and construct them by logsoftmax instead of softmax."""

        ###
        # Randomly initialize emission probabilities.
        # A row for an ordinary tag holds a distribution that sums to 1 over the columns.
        # But EOS_TAG and BOS_TAG have probability 0 of emitting any column's word
        # (instead, they have probability 1 of emitting EOS_WORD and BOS_WORD (respectively), 
        # which don't have columns in this matrix).
        ###

        WB = 0.1*torch.rand(self.k, self.V)  # I added a slightly larger scale to help break initial symmetry

        # bias term to make each tag slightly prefer certain words initially
        for t in range(self.k):
            word_subset = torch.randperm(self.V)[:self.V // self.k]  # assign some words to each tag
            WB[t, word_subset] += 1.0 
        #this is same 
        self.B = WB.softmax(dim=1)            # construct emission distributions p(w | t)
        self.B[self.eos_t, :] = 0             # EOS_TAG can't emit any column's word
        self.B[self.bos_t, :] = 0             # BOS_TAG can't emit any column's word
        
        ###
        # Randomly initialize transition probabilities, in a similar way.
        # Again, we respect the structural zeros of the model.

        #I also tweaked this 
        ###
        
        rows = 1 if self.unigram else self.k
        WA = 0.1*torch.rand(rows, self.k)
        # iagonal bias to encourage some tag self-transitions
        if not self.unigram:
            WA += 0.5 * torch.eye(self.k)[:rows, :]
        
        WA[:, self.bos_t] = float('-inf')  # structural zeros
        
        WA[:, self.bos_t] = -inf    # correct the BOS_TAG column
        self.A = WA.softmax(dim=1)  # construct transition distributions p(t | s)
        if self.unigram:
            # A unigram model really only needs a vector of unigram probabilities
            # p(t), but we'll construct a bigram probability matrix p(t | s) where 
            # p(t | s) doesn't depend on s. 
            # 
            # By treating a unigram model as a special case of a bigram model,
            # we can simply use the bigram code for our unigram experiments,
            # although unfortunately that preserves the O(nk^2) runtime instead
            # of letting us speed up to O(nk) in the unigram case.
            self.A = self.A.repeat(self.k, 1)   # copy the single row k times  

    def printAB(self) -> None:
        """Print the A and B matrices in a more human-readable format (tab-separated)."""
        print("Transition matrix A:")
        col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):   # rows
            row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))
        print("\nEmission matrix B:")        
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):   # rows
            row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")

    def M_step(self, λ: float) -> None:
        """Set the transition and emission matrices (A, B), using the expected
        counts (A_counts, B_counts) that were accumulated by the E step.
        The `λ` parameter will be used for add-λ smoothing.
        We respect structural zeroes ("don't guess when you know")."""

        # # guarding against possible problems
        if λ < 0:
            raise ValueError("Smoothing parameter must be non-negative")
        if not hasattr(self, 'A_counts') or not hasattr(self, 'B_counts'):
            raise RuntimeError("No counts accumulated. Run E_step first.")
        
        # we should have seen no "tag -> BOS" or "BOS -> tag" transitions
        assert self.A_counts[:, self.bos_t].any() == 0, 'Your expected transition counts ' \
                'to BOS are not all zero, meaning you\'ve accumulated them incorrectly!'
        assert self.A_counts[self.eos_t, :].any() == 0, 'Your expected transition counts ' \
                'from EOS are not all zero, meaning you\'ve accumulated them incorrectly!'

        # we should have seen no emissions from BOS or EOS tags
        assert self.B_counts[self.eos_t:self.bos_t, :].any() == 0, 'Your expected emission counts ' \
                'from EOS and BOS are not all zero, meaning you\'ve accumulated them incorrectly!'

        # emission probabilities (this part works well)
        self.B_counts[:self.eos_t] += λ
        row_sums_B = self.B_counts.sum(dim=1, keepdim=True)
        row_sums_B = torch.where(row_sums_B == 0, torch.ones_like(row_sums_B), row_sums_B)
        self.B = self.B_counts / row_sums_B
        self.B[self.eos_t:, :] = 0

        # transition probabilities maybe overkill for normalization
        if self.unigram:
            row_counts = self.A_counts.sum(dim=0) + λ
            WA = torch.log(row_counts + 1e-10).unsqueeze(0)
            WA[:, self.bos_t] = -float('inf')
            self.A = WA.softmax(dim=1)
            self.A = self.A.repeat(self.k, 1)
        else:
            # smoothed copy
            smoothed_A = self.A_counts.clone()
            smoothed_A[:self.eos_t, :] += λ

            # zero out structural zeros before normalization
            smoothed_A[:, self.bos_t] = 0
            smoothed_A[self.eos_t, :] = 0

            row_sums = smoothed_A.sum(dim=1, keepdim=True)
            row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
            self.A = smoothed_A / row_sums
            
        # debugging: print matrices after update
        #print("\nExpected counts A:")
        #print(self.A_counts)
        #print("\nExpected counts B:")
        #print(self.B_counts)

        # debugging :  probabilities sum to 1 where they should
        A_row_sums = self.A[:self.eos_t].sum(dim=1)
        B_row_sums = self.B[:self.eos_t].sum(dim=1)
        assert torch.allclose(A_row_sums, torch.ones_like(A_row_sums), rtol=1e-3), \
            "Transition probabilities don't sum to 1"
        assert torch.allclose(B_row_sums, torch.ones_like(B_row_sums), rtol=1e-3), \
            "Emission probabilities don't sum to 1"
        
    def _zero_counts(self):
        """Set the expected counts to 0.  
        (This creates the count attributes if they didn't exist yet.)"""
        self.A_counts = torch.zeros((self.k, self.k), requires_grad=False)
        self.B_counts = torch.zeros((self.k, self.V), requires_grad=False)

    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[HiddenMarkovModel], float],
              λ: float = 0,
              tolerance: float = 0.001,
              max_steps: int = 50000,
              save_path: Optional[Path|str] = "my_hmm.pkl") -> None:
        """Train the HMM on the given training corpus, starting at the current parameters.
        We will stop when the relative improvement of the development loss,
        since the last epoch, is less than the tolerance.  In particular,
        we will stop when the improvement is negative, i.e., the development loss 
        is getting worse (overfitting).  To prevent running forever, we also
        stop if we exceed the max number of steps."""
        
        if λ < 0:
            raise ValueError(f"{λ=} but should be >= 0")
        elif λ == 0:
            λ = 1e-20
            # Smooth the counts by a tiny amount to avoid a problem where the M
            # step gets transition probabilities p(t | s) = 0/0 = nan for
            # context tags s that never occur at all, in particular s = EOS.
            # 
            # These 0/0 probabilities are never needed since those contexts
            # never occur.  So their value doesn't really matter ... except that
            # we do have to keep their value from being nan.  They show up in
            # the matrix version of the forward algorithm, where they are
            # multiplied by 0 and added into a sum.  A summand of 0 * nan would
            # regrettably turn the entire sum into nan.      
      
        dev_loss = loss(self)   # evaluate the model at the start of training
        
        old_dev_loss: float = dev_loss     # loss from the last epoch
        steps: int = 0   # total number of sentences the model has been trained on so far      
        while steps < max_steps:
            
            # E step: Run forward-backward on each sentence, and accumulate the
            # expected counts into self.A_counts, self.B_counts.
            #
            # Note: If you were using a GPU, you could get a speedup by running
            # forward-backward on several sentences in parallel.  This would
            # require writing the algorithm using higher-dimensional tensor
            # operations, allowing PyTorch to take advantage of hardware
            # parallelism.  For example, you'd update alpha[j-1] to alpha[j] for
            # all the sentences in the minibatch at once (with appropriate
            # handling for short sentences of length < j-1).  

            self._zero_counts()
            for sentence in tqdm(corpus, total=len(corpus), leave=True):
                isent = self._integerize_sentence(sentence, corpus)
                self.E_step(isent)
                steps += 1

            # M step: Update the parameters based on the accumulated counts.
            self.M_step(λ)
            if save_path: self.save(save_path)  # save incompletely trained model in case we crash
            
            # Evaluate with the new parameters
            dev_loss = loss(self)   # this will print its own log messages
            if dev_loss >= old_dev_loss * (1-tolerance):
                # we haven't gotten much better, so perform early stopping
                break
            old_dev_loss = dev_loss            # remember for next eval batch
        
        # Save the trained model.
        if save_path: self.save(save_path)
  
    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> IntegerizedSentence:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""

        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            # Sentence comes from some other corpus that this HMM was not set up to handle.
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")

        return corpus.integerize_sentence(sentence)

    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Compute the log probability of a single sentence under the current
        model parameters.  If the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet.
                
        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're integerizing
        correctly."""

        # Integerize the words and tags of the given sentence, which came from the given corpus.
        isent = self._integerize_sentence(sentence, corpus)
        return self.forward_pass(isent) # (Z(w))

    def E_step(self, isent: IntegerizedSentence, mult: float = 1) -> None:
        """Runs the forward backward algorithm on the given sentence. The forward step computes
        the alpha probabilities.  The backward step computes the beta probabilities and
        adds expected counts to self.A_counts and self.B_counts.  
        
        The multiplier `mult` says how many times to count this sentence. 
        
        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""
        

        #we run the forward pass for alpha and logZ
        # originally this caused me some trouble bc i thought we would have to save something 
        # here, but this is just an internal update !

        self.forward_pass(isent)

        # we run backward for beta and expected counts w the updated values :)
        self.backward_pass(isent, mult)

        # check that the two values are close 
        #if not torch.isclose(self.log_Z, log_Z_backward, atol=1e-5):
            #print(f"Warning: log_Z from forward pass ({self.log_Z.item()}) and backward pass ({log_Z_backward.item()}) do not match.")

        # no need to return anything everything is in the model 

        #for count accum 
        word_ids = torch.tensor([w for w, _ in isent], dtype=torch.long)
        tag_ids = torch.tensor([t if t is not None else -1 for _, t in isent], dtype=torch.long)
        T = len(word_ids) - 2

       # Create mask for valid tags once
        valid_mask = torch.ones(self.k, dtype=torch.bool)
        valid_mask[self.bos_t] = False
        valid_mask[self.eos_t] = False
        valid_indices = torch.where(valid_mask)[0]

        log_A = torch.log(self.A + 1e-10)
        log_B = torch.log(self.B + 1e-10)

        for j in range(1, T + 1):  # skip BOS position
            word_id = word_ids[j]
            tag_id = tag_ids[j]

            # all of these have two cases, for supervised and for unsupervised 
            if tag_id != -1:  # supervised # don't need to use alpha and beta probabilities, there's only one path through the trellis 
                self.B_counts[tag_id, word_id] += mult
                if j < T and tag_ids[j+1] != -1:
                    self.A_counts[tag_id, tag_ids[j+1]] += mult
            else:  # unsupervised # need to use alpha and beta probabilities 
                # emission probabilities
                log_posterior = self.alpha[j, valid_indices] + self.beta[j, valid_indices] - self.log_Z # probability of getting the
                posterior = torch.exp(log_posterior)
                self.B_counts[valid_indices, word_id] += mult * posterior

                # transition probabilities
                if j < T:
                    next_word = word_ids[j + 1]
                    next_tag = tag_ids[j+1]

                    if next_tag != -1:
                        log_posterior = (self.alpha[j, valid_indices] +
                                        log_A[valid_indices, next_tag] +
                                        log_B[next_tag, next_word] +
                                        self.beta[j + 1, next_tag] -
                                        self.log_Z)
                        posterior = torch.exp(log_posterior)
                        self.A_counts[valid_indices, next_tag] += mult * posterior
                    else:  
                        # All valid current tags to all valid next tags
                        for curr_tag in valid_indices:
                            log_posterior = (self.alpha[j, curr_tag] +
                                        log_A[curr_tag, valid_indices] +
                                        log_B[valid_indices, next_word] +
                                        self.beta[j + 1, valid_indices] -
                                        self.log_Z)
                            posterior = torch.exp(log_posterior)
                            self.A_counts[curr_tag, valid_indices] += mult * posterior

        # BOS transitions
        if tag_ids[1] != -1: 
            self.A_counts[self.bos_t, tag_ids[1]] += mult
        else:
            log_posterior = (log_A[self.bos_t, valid_indices] +
                            log_B[valid_indices, word_ids[1]] +
                            self.beta[1, valid_indices] -
                            self.log_Z)
            posterior = torch.exp(log_posterior)
            self.A_counts[self.bos_t, valid_indices] += mult * posterior

        # EOS transitions
        if tag_ids[T] != -1:  
            self.A_counts[tag_ids[T], self.eos_t] += mult
        else:
            log_posterior = self.alpha[T, valid_indices] + log_A[valid_indices, self.eos_t] - self.log_Z
            posterior = torch.exp(log_posterior)
            self.A_counts[valid_indices, self.eos_t] += mult * posterior
    
    @typechecked
    def forward_pass(self, isent: IntegerizedSentence) -> TorchScalar:
        """Run the forward algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the forward
        probability) as a TorchScalar.  If the sentence is not fully tagged, the 
        forward probability will marginalize over all possible tags.  
        
        As a side effect, remember the alpha probabilities and log_Z
        (store some representation of them into attributes of self)
        so that they can subsequently be used by the backward pass."""
        
        # The "nice" way to construct the sequence of vectors alpha[0],
        # alpha[1], ...  is by appending to a List[Tensor] at each step.
        # But to better match the notation in the handout, we'll instead
        # preallocate a list alpha of length n+2 so that we can assign 
        # directly to each alpha[j] in turn.
        self.setup_sentence(isent)
        # extract word IDs excluding BOS and EOS
        word_ids = torch.tensor([w for w, _ in isent[1:-1]], dtype=torch.long)  # exclude BOS and EOS
        T = len(word_ids) + 1  

        alpha = torch.full((T,self.k), float('-inf'))
        # initial alpha for BOS, log(1)
        alpha[0, self.bos_t] = 0.0 
        
        #valid_tags = [t for t in range(self.k) if t != self.bos_t and t != self.eos_t]

        #log_A = torch.log(self.A + 1e-10)
        #log_B = torch.log(self.B + 1e-10)

        #scaling as in other functions
        #scaling_factors  = []
        alphas = [alpha[0]]

        # Forward pass
        for t in range(1, T):
            prev_alpha = alphas[-1].clone()
        
            # Compute alpha[t] for all states
            A = self.A_at(t, isent)
            B = self.B_at(t, isent)
            
            log_A = torch.log(A + 1e-10)
            log_B = torch.log(B + 1e-10)

            alpha_t = torch.logsumexp(prev_alpha.unsqueeze(1) + log_A, dim=0)
            alpha_t = torch.where(torch.tensor([i == self.bos_t for i in range(self.k)]),
                            torch.tensor(float('-inf')),
                            alpha_t)
            #alpha[t] = alpha_t + log_B[:, word_ids[t-1]]
            next_alpha = alpha_t + log_B[:, word_ids[t-1]]
            alphas.append(next_alpha)

            # scaling to prevent underflow
            #max_alpha = torch.max(alpha_t)
            #alpha[t] = alpha_t - max_alpha
            #scaling_factors.append(max_alpha)

        #  alpha for backward pass
        alpha = torch.stack(alphas)
        self.alpha = alpha

        final_A = self.A_at(T, isent)
        self.log_Z = torch.logsumexp(alpha[T-1] + torch.log(final_A + 1e-10)[:, self.eos_t], dim=0)

        #  log probability (log Z) is alpha at EOS position plus scaling
        #self.log_Z = torch.logsumexp(alpha[T-1] + log_A[:, self.eos_t], dim=0)

        #self.scaling_factors = scaling_factors

            # Note: once you have this working on the ice cream data, you may
            # have to modify this design slightly to avoid underflow on the
            # English tagging data. See section C in the reading handout.

        return self.log_Z

    @typechecked
    def backward_pass(self, isent: IntegerizedSentence, mult: float = 1) -> TorchScalar:
        """
        We wanted this to work for supervised, semi-supervised, and unsupervised data."""
        self.setup_sentence(isent)
        
        word_ids = torch.tensor([w for w, _ in isent], dtype=torch.long)
        T = len(word_ids) - 2  # exclude BOS and EOS

        beta = torch.full((T + 2, self.k), float('-inf'))
        beta[-1, self.eos_t] = 0.0

        # pre comp these for faster alg 
        #log_A = torch.log(self.A + 1e-10)
        #log_B = torch.log(self.B + 1e-10)

        #beta = torch.full((T + 2, self.k), float('-inf'))
        #beta[-1, self.eos_t] = 0.0

        # Create mask for valid tags
        valid_mask = torch.ones(self.k, dtype=torch.bool)
        valid_mask[self.bos_t] = False
        valid_mask[self.eos_t] = False
        valid_indices = torch.where(valid_mask)[0]
        
        # Handle T position first (transitions to EOS)
        # Get final transition matrix
        final_A = self.A_at(T+1, isent) 
        beta[T, valid_indices] = torch.log(final_A + 1e-10)[valid_indices, self.eos_t]
    
        #beta[T, valid_indices] = log_A[valid_indices, self.eos_t]
        
        # backward pass with scaling
        for j in range(T, -1, -1):
            next_word = word_ids[j+1]

            # Get position-specific matrices
            A = self.A_at(j+1, isent)
            B = self.B_at(j+1, isent)
            
            log_A = torch.log(A + 1e-10)
            log_B = torch.log(B + 1e-10)

            if j == T:
                beta[j, valid_indices] = log_A[valid_indices, self.eos_t]
            else:
                trans_scores = (log_A[valid_indices][:, valid_indices] + 
                            log_B[valid_indices, next_word].unsqueeze(0) + 
                            beta[j + 1, valid_indices].unsqueeze(0))
                
                beta[j, valid_indices] = torch.logsumexp(trans_scores, dim=1)

            #if j < T :  # don't scale at position 0
               #beta[j] = beta[j] - self.scaling_factors[j]
            
        self.beta = beta 

        return torch.logsumexp(beta[0], dim=0)


    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model."""


        # Note: This code is mainly copied from the forward algorithm.
        # We just switch to using max, and follow backpointers.
        # The code continues to use the name alpha, rather than \hat{alpha}
        # as in the handout.

        # We'll start by integerizing the input Sentence. You'll have to
        # deintegerize the words and tags again when constructing the return
        # value, since the type annotation on this method says that it returns a
        # Sentence object, and that's what downstream methods like eval_tagging
        # will expect.  (Running mypy on your code will check that your code
        # conforms to the type annotations ...)

        isent = self._integerize_sentence(sentence, corpus)
        self.setup_sentence(isent)
        n = len(isent) - 2  # exclude BOS and EOS
        
        word_ids = torch.tensor([w for w, _ in isent[1:-1]], dtype=torch.long)
        #log_A = torch.log(torch.where(self.A > 0, self.A, torch.tensor(1e-10)))
        #log_B = torch.log(torch.where(self.B > 0, self.B, torch.tensor(1e-10)))
        
        # exclusing BOS and EOS valid tag mask  
        valid_mask = torch.ones(self.k, dtype=torch.bool)
        valid_mask[self.bos_t] = False
        valid_mask[self.eos_t] = False
        valid_indices = torch.where(valid_mask)[0]


        
        alpha = torch.full((n + 2, self.k), float('-inf'))
        backpointers = torch.full((n + 2, self.k), -1, dtype=torch.long)

        # (log probability of 1)
        alpha[0, self.bos_t] = 0.0

        # for position 1, first word after BOS handle more efficiently 
        A = self.A_at(1, isent)
        B = self.B_at(1, isent)
        log_A = torch.log(torch.where(A > 0, A, torch.tensor(1e-10)))
        log_B = torch.log(torch.where(B > 0, B, torch.tensor(1e-10)))

        # Handle first word after BOS
        word_id = word_ids[0]
        scores_1 = log_A[self.bos_t, valid_indices] + log_B[valid_indices, word_id]
        alpha[1, valid_indices] = scores_1
        backpointers[1, valid_indices] = self.bos_t

        for j in range(2, n + 1):  # Positions 2 to n
            word_id = word_ids[j-1]

            # Get position-specific matrices
            A = self.A_at(j, isent)
            B = self.B_at(j, isent)
            log_A = torch.log(torch.where(A > 0, A, torch.tensor(1e-10)))
            log_B = torch.log(torch.where(B > 0, B, torch.tensor(1e-10)))
            
        
            # all possible transitions at once [prev_tags, curr_tags]
            scores = (alpha[j-1, valid_indices].unsqueeze(1) +  
                    log_A[valid_indices][:, valid_indices] +   
                    log_B[valid_indices, word_id])            
            
            # best previous tag for each current tag
            max_scores, best_prev = torch.max(scores, dim=0)  # max along previous tags

            alpha[j, valid_indices] = max_scores
            backpointers[j, valid_indices] = valid_indices[best_prev]

        # transition to EOS at position n+1, more efficient
        final_A = self.A_at(n+1, isent)
        log_final_A = torch.log(torch.where(final_A > 0, final_A, torch.tensor(1e-10)))
        final_scores = alpha[n, valid_indices] + log_final_A[valid_indices, self.eos_t]
        max_final_score, best_final = torch.max(final_scores, dim=0)
        alpha[n + 1, self.eos_t] = max_final_score
        backpointers[n + 1, self.eos_t] = valid_indices[best_final]

        # backtracking
        tags = []
        current_tag = self.eos_t
        for j in range(n + 1, 0, -1):
            prev_tag = backpointers[j, current_tag].item()
            if j != n + 1:  # don't include EOS tag
                tags.insert(0, current_tag)
            current_tag = prev_tag

        # now include BOS and EOS
        result = []
        for i, (word, _) in enumerate(sentence):
            if i == 0:  
                result.append((word, BOS_TAG))  # Use constant from corpus.py
            elif i == len(sentence) - 1:
                result.append((word, EOS_TAG))  # Use constant from corpus.py
            else:
                tag_idx = tags[i-1]
                result.append((word, self.tagset[tag_idx]))

        return Sentence(result)
    
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Precompute any quantities needed for forward/backward/Viterbi algorithms.
        This method may be overridden in subclasses."""
        pass

    def save(self, path: Path|str, checkpoint=None, checkpoint_interval: int = 300) -> None:
        """Save this model to the file named by path.  Or if checkpoint is not None, insert its 
        string representation into the filename and save to a temporary checkpoint file (but only 
        do this save if it's been at least checkpoint_interval seconds since the last save).  If 
        the save is successful, then remove the previous checkpoint file, if any."""

        if isinstance(path, str): path = Path(path)   # convert str argument to Path if needed

        now = time.time()
        old_save_time =           getattr(self, "_save_time", None)
        old_checkpoint_path =     getattr(self, "_checkpoint_path", None)
        old_total_training_time = getattr(self, "total_training_time", 0)

        if checkpoint is None:
            self._checkpoint_path = None   # this is a real save, not a checkpoint
        else:    
            if old_save_time is not None and now < old_save_time + checkpoint_interval: 
                return   # we already saved too recently to save another temp version
            path = path.with_name(f"{path.stem}-{checkpoint}{path.suffix}")  # use temp filename
            self._checkpoint_path = path

        
        # Save the model with the fields set as above, so that we'll 
        # continue from it correctly when we reload it.
        try:
            torch.save(self, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved model to {path}")
        except Exception as e:   
            # something went wrong with the save; so restore our old fields,
            # so that caller can potentially catch this exception and try again
            self._save_time          = old_save_time
            self._checkpoint_path    = old_checkpoint_path
            self.total_training_time = old_total_training_time
            raise e
        
        # Since save was successful, remember it and remove old temp version (if any)
        self._save_time = now
        if old_checkpoint_path: 
            try: os.remove(old_checkpoint_path)
            except FileNotFoundError: pass  # don't complain if the user already removed it manually


    @classmethod
    def load(cls, path: Path|str, device: str = 'cpu') -> HiddenMarkovModel:
        if isinstance(path, str): path = Path(path)   # convert str argument to Path if needed
            
        # torch.load is similar to pickle.load but handles tensors too
        # map_location allows loading tensors on different device than saved
        model = torch.load(path, map_location=device)

        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls.__name__} but got {model.__class__.__name__} " \
                             f"from saved file {path}.")

        logger.info(f"Loaded model from {path}")
        return model

    @typechecked
    def posterior_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """find the best tag for each position with posterior marginal probs."""
        isent = self._integerize_sentence(sentence, corpus)
        n = len(isent) - 2  # exclude BOS and EOS
        
        self.forward_pass(isent)
        self.backward_pass(isent)
        
        valid_mask = torch.ones(self.k, dtype=torch.bool)
        valid_mask[self.bos_t] = False
        valid_mask[self.eos_t] = False
        valid_indices = torch.where(valid_mask)[0]
        
        # find tag with highest posterior probability
        tags = []
        for j in range(1, n+1):  # skip bos and include up to last real word
            log_posterior = self.alpha[j, valid_indices] + self.beta[j, valid_indices] - self.log_Z
            
            # get most probable tag
            best_tag_idx = torch.argmax(log_posterior)
            best_tag = valid_indices[best_tag_idx]
            tags.append(best_tag)
        
        result = []
        for i, (word, _) in enumerate(sentence):
            if i == 0:
                result.append((word, BOS_TAG))  # Use constant from corpus.py
            elif i == len(sentence) - 1:
                result.append((word, EOS_TAG))  # Use constant from corpus.py
            else:
                tag_idx = tags[i-1]
                result.append((word, self.tagset[tag_idx]))
        
        return Sentence(result)

@typechecked
class EnhancedHMM(HiddenMarkovModel):
    """ Decided to do the improvements this way because there were a few methods that just started looking
    overgrown for lack of a better word """
    
    def __init__(self, tagset: Integerizer[Tag], 
                vocab: Integerizer[Word], 
                 unigram: bool = False, 
                 supervised_constraint: bool = True,
                 better_smoothing: bool = True):
        super().__init__(tagset, vocab, unigram)
        self.supervised_constraint = supervised_constraint
        self.better_smoothing = better_smoothing
        self.tag_word_counts = defaultdict(set)  # allowed tags per word
        self.closed_class_tags = set()  # closed class tags
        self.open_class_threshold = 5

    def train(self, corpus: TaggedCorpus, *args, **kwargs):
        """we extended this method to learn tag constraints from supervised data. 
        So unfortunately this wont do too much for our purely unsupervised case, but it's really impressive for the others """
        
        # learn word-tag associations 
        for sentence in corpus:
            isent = self._integerize_sentence(sentence, corpus)
            for word_id, tag_id in isent:
                if tag_id is not None:
                    self.tag_word_counts[word_id].add(tag_id)

        # find tags that only appear with a small vocab
        tag_vocab_sizes = defaultdict(set)
        for word_id, tag_ids in self.tag_word_counts.items():
            for tag_id in tag_ids:
                tag_vocab_sizes[tag_id].add(word_id)

        # mark tags as closed if they appear with few words
        for tag_id, vocab in tag_vocab_sizes.items():
            if len(vocab) < self.open_class_threshold:
                self.closed_class_tags.add(tag_id)

        super().train(corpus, *args, **kwargs)

    def M_step(self, λ: float = 0.01) -> None:
        """set the transition and emission matrices with bounds checking for vocabulary."""
        if λ < 0:
            raise ValueError("Smoothing parameter must be non-negative")
        if not hasattr(self, 'A_counts') or not hasattr(self, 'B_counts'):
            raise RuntimeError("No counts accumulated. Run E_step first.")

        # Verify structural zeros agaaiiiiin
        assert self.A_counts[:, self.bos_t].any() == 0
        assert self.A_counts[self.eos_t, :].any() == 0
        assert self.B_counts[self.eos_t:self.bos_t, :].any() == 0

        if self.better_smoothing:
            #  smoothing matrix - varies by tag type
            B_smoothing = torch.full((self.k, self.V), λ)
            for tag_id in self.closed_class_tags:
                B_smoothing[tag_id, :] = λ * 0.1
            smoothed_B = self.B_counts + B_smoothing
            
            #  supervised constraints if enabled
            if self.supervised_constraint:
                mask = torch.zeros((self.k, self.V), dtype=torch.bool)
                for word_id, tag_ids in self.tag_word_counts.items():
                    # skip words that are outside our vocabulary size
                    if word_id >= self.V:
                        continue
                    for tag_id in tag_ids:
                        if tag_id >= self.k:
                            continue
                        mask[tag_id, word_id] = True
                smoothed_B = torch.where(mask, smoothed_B, torch.zeros_like(smoothed_B))
        else:
            # simple as the fallback case
            smoothed_B = self.B_counts.clone()
            smoothed_B[:self.eos_t] += λ

        # norm emission 
        row_sums_B = smoothed_B.sum(dim=1, keepdim=True)
        row_sums_B = torch.where(row_sums_B == 0, torch.ones_like(row_sums_B), row_sums_B)
        self.B = smoothed_B / row_sums_B
        self.B[self.eos_t:, :] = 0

        # handle transitions
        if self.unigram:
            row_counts = self.A_counts.sum(dim=0) + λ
            WA = torch.log(row_counts + 1e-10).unsqueeze(0)
            WA[:, self.bos_t] = -float('inf')
            self.A = WA.softmax(dim=1)
            self.A = self.A.repeat(self.k, 1)
        else:
            # transition smoothing matrix 
            A_smoothing = torch.full((self.k, self.k), λ)
            A_smoothing[:, self.bos_t] = 0
            A_smoothing[self.eos_t, :] = 0
        
            smoothed_A = self.A_counts + A_smoothing

            smoothed_A[:, self.bos_t] = 0
            smoothed_A[self.eos_t, :] = 0
            
            # norming
            row_sums_A = smoothed_A.sum(dim=1, keepdim=True)
            row_sums_A = torch.where(row_sums_A == 0, torch.ones_like(row_sums_A), row_sums_A)
            self.A = smoothed_A / row_sums_A

        # Verify probabilities sum to 1
        A_row_sums = self.A[:self.eos_t].sum(dim=1)
        B_row_sums = self.B[:self.eos_t].sum(dim=1)
        assert torch.allclose(A_row_sums, torch.ones_like(A_row_sums), rtol=1e-3)
        assert torch.allclose(B_row_sums, torch.ones_like(B_row_sums), rtol=1e-3)
    
    def decode(self, sentence: Sentence, corpus: TaggedCorpus, method: str = 'viterbi') -> Sentence:
        """picks best tags for a sentence. can use viterbi, posterior, or hybrid method.
        hybrid uses constraints for known words and posterior for unknowns - usually works best."""
        
        if method == 'viterbi':
            return self.viterbi_tagging(sentence, corpus)
        elif method == 'posterior':
            return self.posterior_tagging(sentence, corpus)
        elif method == 'hybrid':
            # we got inspired by the mix of training files so this will use 
            # constraints for known words, posterior for unknown
            isent = self._integerize_sentence(sentence, corpus)
            self.forward_pass(isent)
            self.backward_pass(isent)
            
            tags = []
            for j, (word_id, _) in enumerate(isent[1:-1], 1):
                if word_id in self.tag_word_counts:
                    # for known words, only consider tags we've seen before
                    allowed_tags = list(self.tag_word_counts[word_id])
                    log_probs = (self.alpha[j, allowed_tags] + 
                            self.beta[j, allowed_tags] - 
                            self.log_Z)  
                    best_idx = torch.argmax(log_probs)
                    tags.append(allowed_tags[best_idx])
                else:
                    valid_mask = torch.ones(self.k, dtype=torch.bool)
                    valid_mask[self.bos_t] = False
                    valid_mask[self.eos_t] = False
                    valid_indices = torch.where(valid_mask)[0]
                    # for unknown words, use posterior over all tags
                    log_probs = self.alpha[j, valid_indices] + self.beta[j, valid_indices] - self.log_Z
                    best_idx = torch.argmax(log_probs)
                    tags.append(valid_indices[best_idx])
        
            result = []
            for i, (word, _) in enumerate(sentence):
                if i == 0:  # BOS
                    result.append((word, BOS_TAG))
                elif i == len(sentence) - 1:  # EOS
                    result.append((word, EOS_TAG))
                else:
                    # Convert tag index to actual tag using tagset
                    tag_idx = tags[i-1]  # -1 because we don't include BOS
                    result.append((word, self.tagset[tag_idx]))
            
            return Sentence(result)
        else:
            raise ValueError(f"Unknown decoding method: {method}")