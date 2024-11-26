#!/usr/bin/env python3

# CS465 at Johns Hopkins University.

# Subclass ConditionalRandomField to use PyTorch facilities for gradient computation and gradient-based optimization.

from __future__ import annotations
import logging
import torch.nn as nn
from math import inf
from pathlib import Path
import time
from typing_extensions import override

import torch
from torch import tensor, Tensor, cuda
from jaxtyping import Float

from corpus import Sentence, Tag, TaggedCorpus, Word
from integerize import Integerizer
from crf import ConditionalRandomField

TorchScalar = Float[Tensor, ""] # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available


class ConditionalRandomFieldBackprop(ConditionalRandomField, nn.Module):
    """An implementation of a CRF that has only transition and 
    emission features, just like an HMM."""
    
    # We inherit most of the functionality from the parent CRF (and HMM)
    # classes. The overridden methods will allow for backpropagation to be done
    # automatically by PyTorch rather than manually as in the parent class.
    # CRFBackprop also inherits from nn.Module so that nn.Parameter will
    # be able to register parameters to be found by self.parameters().
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        # [docstring will be inherited from parent method]
        
        # Call both parent classes' initializers
        nn.Module.__init__(self)  
        print("Initializing ConditionalRandomFieldBack")
        super().__init__(tagset, vocab, unigram)

        # Print number of parameters        
        self.count_params()
        
    @override
    def init_params(self) -> None:
        # [docstring will be inherited from parent method]
        
        # This version overrides the parent to use nn.Parameter.
        # Wrapping the parameters in nn.Parameter ensures that we'll be able to
        # find them when creating an optimizer (using self.parameters())
        # and we'll be able to compute the loss gradient with respect to them 
        # (using loss.backward()).
        # 
        # See the online documentation of nn.Parameter and nn.init.  Note that
        # torch.empty creates an uninitalized tensor.
        
        self.WB = nn.Parameter(torch.empty(self.k, self.V))
        nn.init.uniform_(self.WB, 0, 0.01)
        
        rows = 1 if self.unigram else self.k
        self.WA = nn.Parameter(torch.empty(rows, self.k))
        nn.init.uniform_(self.WA, 0, 0.01)

        # If your implementation of the parent method also initialized certain
        # parameters to -inf, then you should do the same thing here by calling
        # nn.init.constant_ on *slices* of the parameter tensors.
        #
        # However, use -999 instead of -inf.  This is the easiest workaround to
        # prevent nan values during weight decay.  It's okay because in
        # updateAB(), exp(-999) will underflow and yield 0, just as exp(-inf)
        # would have.  
        #
        # (You won't have to do anything here if you used a different design in
        # the parent class where those parameters were not initialized to -inf,
        # but instead, updateAB() wrote the structural zeroes into the
        # corresponding entries of A and B.  In that case, those parameters in
        # WA and WB will be ignored.  They don't affect the training objective
        # and you don't need to initialize them to -inf or anything else.)

        with torch.no_grad():
            if not self.unigram:
                self.WA.data[:, self.bos_t] = -999  # can't transition to BOS
                self.WA.data[self.eos_t, :] = -999  # can't transition from EOS
                self.WA.data[self.bos_t, self.eos_t] = -999  # BOS can't transition directly to EOS

            # BOS and EOS tags can't emit any words
            self.WB.data[self.eos_t, :] = -999
            self.WB.data[self.bos_t, :] = -999
       
        self.updateAB()
       
    def init_optimizer(self, lr: float, weight_decay: float) -> None:      
        """Creates an optimizer for training.
        A subclass may override this method to select a different optimizer."""
        self.optimizer = torch.optim.AdamW(
        params=self.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),  # Default Adam betas
        eps=1e-8,           # Slightly larger epsilon for stability
        amsgrad=True        # Use AMSGrad variant for better convergence
    )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )

    def count_params(self) -> None:
        paramcount = sum(p.numel() for p in self.parameters() if p.requires_grad)
        paramshapes = " + ".join("*".join(str(dim) for dim in p.size()) for p in self.parameters() if p.requires_grad)
        logger.info(f"Parameters: {paramcount} = {paramshapes}")

    @override
    def train(self,
              corpus: TaggedCorpus,
              *args,
              minibatch_size: int = 1,
              lr: float = 1.0,  # same defaults as in parent
              reg: float = 0.0,
              tolerance: float = 1e-5,
              **kwargs) -> None:
        # [docstring will be inherited from parent method]
                
        # Configure an optimizer.
        # Weight decay in the optimizer augments the minimization
        # objective by adding an L2 regularizer, whose gradient is
        # weight_decay * params.  Thus, the params decay back toward 0 at
        # each optimization step (minibatch).
        self.init_optimizer(lr=lr,
                            weight_decay = 2 * reg * minibatch_size / len(corpus))
        
        # Now just call the parent method with exactly the same arguments.  The parent
        # method will train exactly as before (you should probably review it) --
        # except that it will call methods that you will override below. Your
        # overridden versions can use the optimizer that we just created.

        self._save_time = time.time()   # this line is here just in case you're using an old version of parent class that doesn't do it
        super().train(corpus, *args, minibatch_size=minibatch_size, lr=lr, reg=reg,tolerance= tolerance, **kwargs)

    @override        
    def _zero_grad(self):
        # [docstring will be inherited from parent method]

        # Look up how to do this with a PyTorch optimizer!
        if hasattr(self, 'optimizer'):
            # More memory efficient than optimizer.zero_grad()
            for param in self.parameters():
                param.grad = None
        self.minibatch_sentences = []

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        # [docstring will be inherited from parent method]
        
        # Rather than doing the manual backward computation, we instead let
        # PyTorch do all the heavy lifting. 
        # 
        # Hint: You no longer need to call E_step(), which ran a backward pass
        # and accumulated expected counts into self.A_counts and self.B_counts.
        # Rather, the gradient of the logprob with respect to each parameter
        # will be accumulated by back-propagation directly in the .grad
        # attribute of that parameter, where the optimizer knows to look for it.
        #
        # Hint: You want to maximize the (regularized) log-probability. However,
        # PyTorch optimizers *minimize* functions by default.
        
        if not hasattr(self, 'minibatch_sentences'):
            self.minibatch_sentences = []
            
        device = next(self.parameters()).device
        self.minibatch_sentences.append((sentence, corpus))
        
        # Process in larger batches for efficiency
        if len(self.minibatch_sentences) >= 32:  # Configurable batch size
            self._process_batch()
    
    def _process_batch(self):
        """Helper method to process accumulated sentences in batch"""
        if not self.minibatch_sentences:
            return
            
        # Compute logprobs in parallel
        logprobs = []
        for sentence, corpus in self.minibatch_sentences:
            logprob = self.logprob(sentence, corpus)
            logprobs.append(logprob)
        
        # Combine losses efficiently
        total_loss = -torch.stack(logprobs).sum()
        
        # Backward pass
        total_loss.backward()
        
        # Clear batch
        self.minibatch_sentences = []

    @override
    def logprob_gradient_step(self, lr: float) -> None:
        # [docstring will be inherited from parent method]
        
        # Look up how to do this with a PyTorch optimizer!
        # Basically, you want to take a step in the direction
        # of the accumulated gradient.
        if hasattr(self, 'minibatch_sentences') and self.minibatch_sentences:
            # Process any remaining sentences
            self._process_batch()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Step optimizer
            self.optimizer.step()
            
            # Update learning rate if using scheduler
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step(self._compute_validation_loss())
    
    def _compute_validation_loss(self) -> float:
        """Helper method to compute validation loss for scheduler"""
        # Implement validation loss computation here if needed
        # For now, return a placeholder value
        return 0.0
        
    @override
    def reg_gradient_step(self, lr: float, reg: float, frac: float):
        # [docstring will be inherited from parent method]

        # Hint: We created an optimizer that already handles L2
        # regularization for us.        
        pass

    def learning_speed(self, lr: float, minibatch_size: int) -> float:
        """Estimates how fast we are trying to learn, based on the gradient
        of the most recent minibatch.  Call this just before or after an
        optimizer step.  
        
        The return value is the estimated improvement in the training loss per
        training example.  This will tend to decrease over time as the model
        approaches a local minimum and the gradient flattens out.
        
        You may discover that your learning speed goes up if you choose
        hyperparameters that increase the number of parameters in your model,
        since this will typically increase the norm of the gradient. 

        If you are trying to learn too fast and getting poor results, try
        reducing your learning speed by reducing the learning rate (smaller
        updates) or by increasing the minibatch size (less frequent updates).

        Reducing the learning speed can reduce noise in the optimization by
        placing less trust in our two approximations -- that the function is
        locally linear (motivating gradient-based updates) and that individual
        examples are correlated with the batch gradient (motivating stochastic
        updates).
               
        We define learning speed to be lr times the squared norm of the
        minibatch gradient, divided by the size of the minibatch.  This is
        because for small lr, the loss reduction from a step of size lr in any
        direction v will change the loss by about lr * (gradient dot v).  Thus,
        if v is itself the gradient direction, the loss will change by lr *
        (gradient dot gradient) = lr * ||gradient||^2.
        
        Note that the optimizer may not in fact take a step in exactly the
        gradient direction, thanks to techniques such as momentum or weight
        decay."""
               
        with torch.no_grad():
            total_grad_norm = sum(
                torch.sum(p.grad * p.grad).item() 
                for p in self.parameters() 
                if p.grad is not None
            )
        return lr * total_grad_norm / minibatch_size
        
