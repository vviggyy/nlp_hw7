#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Starter code for Conditional Random Fields.

from __future__ import annotations
import logging
from math import inf, log, exp
from pathlib import Path
from typing import Callable, Optional
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float

import itertools, more_itertools
from tqdm import tqdm # type: ignore

from corpus import (BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag,
                    TaggedCorpus, Word)
from integerize import Integerizer
from hmm import HiddenMarkovModel

TorchScalar = Float[Tensor, ""] # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomField(HiddenMarkovModel):
    """An implementation of a CRF that has only transition and 
    emission features, just like an HMM."""
    
    # CRF inherits forward-backward and Viterbi methods from the HMM parent class,
    # along with some utility methods.  It overrides and adds other methods.
    # 
    # Really CRF and HMM should inherit from a common parent class, TaggingModel.  
    # We eliminated that to make the assignment easier to navigate.
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        """Construct an CRF with initially random parameters, with the
        given tagset, vocabulary, and lexical features.  See the super()
        method for discussion."""

        super().__init__(tagset, vocab, unigram)

    @override
    def init_params(self) -> None:
        """Initialize params self.WA and self.WB to small random values, and
        then compute the potential matrices A, B from them.
        As in the parent method, we respect structural zeroes ("Don't guess when you know")."""

        # See the "Training CRFs" section of the reading handout.
        # 
        # For a unigram model, self.WA should just have a single row:
        # that model has fewer parameters.

        ###
        # Randomly initialize emission probabilities.
        # A row for an ordinary tag holds a distribution that sums to 1 over the columns.
        # But EOS_TAG and BOS_TAG have probability 0 of emitting any column's word
        # (instead, they have probability 1 of emitting EOS_WORD and BOS_WORD (respectively), 
        # which don't have columns in this matrix).
        ###

        #copied from hmm init_params so i used the same multiplication factor 
        #EMISSIONS
        k = len(self.tagset)
        v = len(self.vocab) - 2

        if self.unigram:
            # tag uniform params
            self.WA = torch.randn(1,k) * 0.1
        else:
            # tag bigram params
            self.WA = torch.randn(k, k) * 0.1
            self.WA[:, self.bos_t] = float('-inf')  # can't transition to BOS
            self.WA[self.eos_t, :] = float('-inf') # cant transition from EOS
            self.WA[self.bos_t, self.eos_t] = float('-inf') # BOS can't transition directly to EOS
            
        # initialize emission parameters
        self.WB = torch.randn(k, v) * 0.1


        # BOS and EOS tags can't emit any words
        self.WB[self.eos_t, :] = float('-inf')
        self.WB[self.bos_t, :] = float('-inf')

            

        ###
        # Randomly initialize transition probabilities, in a similar way.
        # Again, we respect the structural zeros of the model.
        ###
        '''
        changed this part a little 
        rows = 1 if self.unigram else self.k
        self.WA = 0.01*torch.rand(rows, self.k) #kxk
        self.WA[:, self.bos_t] = -1e10    # correct the BOS_TAG column
        '''

        #self.A = self.WA.softmax(dim=1)  # construct transition distributions p(t | s)
        #if self.unigram:
            # A unigram model really only needs a vector of unigram probabilities
            # p(t), but we'll construct a bigram probability matrix p(t | s) where 
            # p(t | s) doesn't depend on s. 
            # 
            # By treating a unigram model as a special case of a bigram model,
            # we can simply use the bigram code for our unigram experiments,
            # although unfortunately that preserves the O(nk^2) runtime instead
            # of letting us speed up to O(nk) in the unigram case.
        #    self.A = self.A.repeat(self.k, 1)   # copy the single row k times
            
        self.updateAB()   # compute potential matrices

    def updateAB(self) -> None:
        """Set the transition and emission matrices self.A and self.B, 
        based on the current parameters self.WA and self.WB.
        See the "Parametrization" section of the reading handout."""
       
        # Even when self.WA is just one row (for a unigram model), 
        # you should make a full k × k matrix A of transition potentials,
        # so that the forward-backward code will still work.
        # See init_params() in the parent class for discussion of this point.
        if self.unigram:
            self.A = torch.exp(self.WA).expand(self.k, -1).clone()
        else:
            self.A = torch.exp(self.WA).clone()
        
        self.B = torch.exp(self.WB).clone()
        # need some way for WA, WB to be updated in the first place... 
        
    @override
    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[ConditionalRandomField], float],
              tolerance: float =0.001,
              minibatch_size: int = 1,
              eval_interval: int = 500,
              lr: float = 1.0,
              reg: float = 0.0,
              max_steps: int = 50000,
              save_path: Optional[Path] = Path("my_hmm.pkl")) -> None:
        """Train the CRF on the given training corpus, starting at the current parameters.

        The minibatch_size controls how often we do an update.
        (Recommended to be larger than 1 for speed; can be inf for the whole training corpus,
        which yields batch gradient ascent instead of stochastic gradient ascent.)
        
        The eval_interval controls how often we evaluate the loss function (which typically
        evaluates on a development corpus).
        
        lr is the learning rate, and reg is an L2 batch regularization coefficient.

        We always do at least one full epoch so that we train on all of the sentences.
        After that, we'll stop after reaching max_steps, or when the relative improvement 
        of the evaluation loss, since the last evalbatch, is less than the
        tolerance.  In particular, we will stop when the improvement is
        negative, i.e., the evaluation loss is getting worse (overfitting)."""
        
        def _loss() -> float:
            # Evaluate the loss on the current parameters.
            # This will print its own log messages.
            # 
            # In the next homework we will extend the codebase to use backprop, 
            # which finds gradient with respect to the parameters.
            # However, during evaluation on held-out data, we don't need this
            # gradient and we can save time by turning off the extra bookkeeping
            # needed to compute it.
            with torch.no_grad():  # type: ignore 
                return loss(self)      

        # did not chnage this 

        # This is relatively generic training code.  Notice that the
        # updateAB() step before each minibatch produces A, B matrices
        # that are then shared by all sentences in the minibatch.
        # 
        # All of the sentences in a minibatch could be treated in
        # parallel, since they use the same parameters.  The code
        # below treats them in series -- but if you were using a GPU,
        # you could get speedups by writing the forward algorithm
        # using higher-dimensional tensor operations that update
        # alpha[j-1] to alpha[j] for all the sentences in the
        # minibatch at once.  PyTorch could then take better advantage
        # of hardware parallelism on the GPU.

        if reg < 0: raise ValueError(f"{reg=} but should be >= 0")
        if minibatch_size <= 0: raise ValueError(f"{minibatch_size=} but should be > 0")
        if minibatch_size > len(corpus):
            minibatch_size = len(corpus)  # no point in having a minibatch larger than the corpus
        min_steps = len(corpus)   # always do at least one epoch

        #self.init_params()    # initialize the parameters and call updateAB()
        self._zero_grad()     # get ready to accumulate their gradient
        steps = 0
        old_loss = _loss()    # evaluate initial loss
        for evalbatch in more_itertools.batched(
                           itertools.islice(corpus.draw_sentences_forever(), 
                                            max_steps),  # limit infinite iterator
                           eval_interval): # group into "evaluation batches"
            for sentence in tqdm(evalbatch, total=eval_interval):
                # Accumulate the gradient of log p(tags | words) on this sentence 
                # into A_counts and B_counts.
                self.accumulate_logprob_gradient(sentence, corpus)
                steps += 1
                
                if steps % minibatch_size == 0:              
                    # Time to update params based on the accumulated 
                    # minibatch gradient and regularizer.
                    self.logprob_gradient_step(lr)
                    self.reg_gradient_step(lr, reg, minibatch_size / len(corpus))
                    self.updateAB()      # update A and B potential matrices from new params
                    if save_path: self.save(save_path, checkpoint=steps)  
                    self._zero_grad()    # get ready to accumulate a new gradient for next minibatch
            
            # Evaluate our progress.
            curr_loss = _loss()
            if steps >= min_steps and curr_loss >= old_loss * (1-tolerance):
                break   # we haven't gotten much better since last evalbatch, so stop
            old_loss = curr_loss   # remember for next evalbatch

        # For convenience when working in a Python notebook, 
        # we automatically save our training work by default.
        if save_path: self.save(save_path)
 
    @override
    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Return the *conditional* log-probability log p(tags | words) under the current
        model parameters.  This behaves differently from the parent class, which returns
        log p(tags, words).
        
        Just as for the parent class, if the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  Note that if the sentence is completely
        untagged, then the marginal probability will be 1.
                
        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're integerizing
        correctly."""

        # Integerize the words and tags of the given sentence, which came from the given corpus.
        #isent = self._integerize_sentence(sentence, corpus)

        # Remove all tags and re-integerize the sentence.
        # Working with this desupervised version will let you sum over all taggings
        # in order to compute the normalizing constant for this sentence.
        #desup_isent = self._integerize_sentence(sentence.desupervise(), corpus)

        #basically the same thing as given but doing the integerizing elsewhere
        numerator = super().logprob(sentence, corpus)
        
        # Get log Z(w) = log ∑_t p(t,w) from untagged sequence
        denominator = super().logprob(sentence.desupervise(), corpus)
    
        
        return numerator - denominator

    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        """Add the gradient of self.logprob(sentence, corpus) into a total minibatch
        gradient that will eventually be used to take a gradient step."""
        
        # In the present class, the parameters are self.WA, self.WB, the gradient
        # is a difference of observed and expected counts, and you'll accumulate
        # the gradient information into self.A_counts and self.B_counts.  
        # 
        # (In the next homework, you'll have fancier parameters and a fancier gradient,
        # so you'll override this and accumulate the gradient using PyTorch's
        # backprop instead.)
        
        # Just as in logprob()
        isent_tagged   = self._integerize_sentence(sentence, corpus)
        isent_untagged = self._integerize_sentence(sentence.desupervise(), corpus)

        # Hint: use the mult argument to E_step(). <-- clever!!
        
        self.E_step(isent_tagged, mult=1.0) #obs counts from tagged
    
        self.E_step(isent_untagged, mult=-1.0) # subtract exp counts from untagged

            
    def _zero_grad(self):
        """Reset the gradient accumulator to zero."""
        # You'll have to override this method in the next homework; 
        # see comments in accumulate_logprob_gradient().
        self._zero_counts()

    def logprob_gradient_step(self, lr: float) -> None:
        """Update the parameters using the accumulated logprob gradient.
        lr is the learning rate (stepsize)."""
        
        # Warning: Careful about how to handle the unigram case, where self.WA
        # is only a vector of tag unigram potentials (even though self.A_counts
        # is a still a matrix of tag bigram potentials).
        
        #copied from HMM init_params 
        if self.unigram:
            # A unigram model really only needs a vector of unigram probabilities
            # p(t), but we'll construct a bigram probability matrix p(t | s) where 
            # p(t | s) doesn't depend on s. 
            # 
            # By treating a unigram model as a special case of a bigram model,
            # we can simply use the bigram code for our unigram experiments,
            # although unfortunately that preserves the O(nk^2) runtime instead
            # of letting us speed up to O(nk) in the unigram case.


            #sum over previous tags
            self.WA += lr * self.A_counts.sum(dim =0) # copy the single row k times 
        else:
            self.WA += lr * self.A_counts
            
        # update parameters
        # self.WA += self.A_counts
        self.WB += lr * self.B_counts
        #raise NotImplementedError   # you fill this in!
        
    def reg_gradient_step(self, lr: float, reg: float, frac: float):
        """Update the parameters using the gradient of our regularizer.
        More precisely, this is the gradient of the portion of the regularizer 
        that is associated with a specific minibatch, and frac is the fraction
        of the corpus that fell into this minibatch."""
                    
        # Because this brings the weights closer to 0, it is sometimes called
        # "weight decay".
        
        if reg == 0: return      # can skip this step if we're not regularizing

        # Weight decay factor for this minibatch
        decay = 1 - 2 * lr * reg * frac  
        
        # Only decay finite parameters
        if self.unigram:
            mask = torch.isfinite(self.WA)
            self.WA[mask] *= decay
        else:
            mask = torch.isfinite(self.WA)
            self.WA[mask] *= decay
            
        mask = torch.isfinite(self.WB) 
        self.WB[mask] *= decay

        # Warning: Be careful not to do something like w -= 0.1*w,
        # because some of the weights are infinite and inf - inf = nan. 
        # Instead, you want something like w *= 0.9.
        
        #self.WA -= (-1 * lr *  reg / frac) * self.A
        #self.WB -= (-1 * lr *  reg / frac) * self.B
        #adjust by gradient
    
    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the current model."""
        # use the parent's viterbi
        return super().viterbi_tagging(sentence, corpus)
    def posterior_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the current model."""
        # use the parent's posterior
        return super().posterior_tagging(sentence, corpus)
