#!/usr/bin/env python3
"""
Command-line interface for training and evaluating HMM and CRF taggers.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import logging
from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
import torch
from eval import model_cross_entropy, viterbi_error_rate, write_tagging
from hmm import HiddenMarkovModel, EnhancedHMM
from crf import ConditionalRandomField
from corpus import TaggedCorpus
from corpus import (Sentence, BOS_TAG, EOS_TAG, BOS_WORD, EOS_WORD, 
                   OOV_WORD, TaggedCorpus)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    ###
    # HW6 and HW7 - General shared training parameters
    ###

    filegroup = parser.add_argument_group("Model and data files")

    filegroup.add_argument("input", type=str, help="input sentences for evaluation (labeled dev set or test set)")

    filegroup.add_argument(
        "-m",
        "--model",
        type=str,
        help="file where the model will be saved.  If it already exists, it will be loaded; otherwise a randomly initialized model will be created"
    )

    filegroup.add_argument(
        "-t",
        "--train",
        type=str,
        nargs="*",
        default=[],
        help="optional training data files to train the model further"
    )

    filegroup.add_argument(
        "-o",
        "--output_file",
        type=str,
        default=None,
        help="where to save the prediction outputs"
    )

    traingroup = parser.add_argument_group("Training procedure")

    traingroup.add_argument(
        "--loss",
        type=str,
        default="cross_entropy",
        choices=['cross_entropy','viterbi_error'],
        help="loss function to evaluate on during training and final evaluation"
    )

    traingroup.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="tolerance for detecting convergence of loss function during training"
    )

    traingroup.add_argument(
        "--max_steps",
        type=int,
        default=50000,
        help="maximum number of training steps (measured in sentences, not epochs or minibatches)"
    )

    modelgroup = parser.add_argument_group("Tagging model structure")

    modelgroup.add_argument(
        "-u",
        "--unigram",
        action="store_true",
        default=False,
        help="model should be only a unigram HMM or CRF (baseline)"
    )

    #for extra cred
    modelgroup.add_argument(
        "--decoder",
        type=str,
        default="viterbi",
        choices=['viterbi', 'posterior'],
        help="decoding method to use (viterbi or posterior)"
    )
    
    modelgroup.add_argument(
        "--crf",
        action="store_true",
        default=False,
        help="model should be a CRF rather than an HMM"
    )

    modelgroup.add_argument(
        "-l",
        "--lexicon",
        type=str,
        default=None,
        help="model should use word embeddings drawn from this lexicon" 
    )
    
    modelgroup.add_argument(
        "-a",
        "--awesome",
        action="store_true",
        default=False,
        help="model should use extra improvements"
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
    )

    hmmgroup = parser.add_argument_group("HMM-specific options (ignored for CRF)")

    hmmgroup.add_argument(
        "--lambda",
        dest="λ",
        type=float,
        default=0,
        help="lambda for add-lambda smoothing in the HMM M-step"
    )

    crfgroup = parser.add_argument_group("CRF-specific options (ignored for HMM)")

    crfgroup.add_argument(
        "--reg",
        type=float,
        default=0.0,
        help="l2 regularization coefficient during training"
    )

    crfgroup.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="learning rate during CRF training"
    )


    crfgroup.add_argument(
        "--batch_size",
        type=int,
        default=30,
        help="mini-batch size: number of training sentences per gradient update"
    )

    crfgroup.add_argument(
        "--eval_interval",
        type=int,
        default=2000,
        help="how often to evaluate the model (after training on this many sentences)"
    )

    crfgroup.add_argument(
        "-r",
        "--rnn_dim",
        type=int,
        default=None,
        help="model should encode context using recurrent neural nets with this hidden-state dimensionality (>= 0)"
    )

    awesomegroup = parser.add_argument_group("Awesome mode options")

    awesomegroup.add_argument(
        "--awesome_decoder",
        type=str,
        default="hybrid",
        choices=['viterbi', 'posterior', 'hybrid'],
        help="decoding method to use"
    )
    awesomegroup.add_argument(
        "--supervised_constraint",
        action="store_true",
        default=True,
        help="use supervised tag constraints"
    )
    awesomegroup.add_argument(
        "--smart_smoothing",
        action="store_true",
        default=True,
        help="use differential smoothing based on tag characteristics"
    )
    awesomegroup.add_argument(
        "--optimize_hyperparams",
        action="store_true",
        help="run hyperparameter optimization"
    )

    args = parser.parse_args()

    ### Any arg manipulation and checking goes here

    # Get paths where we'll load and save model (possibly none).
    # These are added to the args namespace.
    if args.model is None:
        args.load_path = args.save_path = None
    else:
        args.load_path = args.save_path = Path(args.model)
        if not args.load_path.exists(): args.load_path = None  # only save here

    # Default path where we'll save the outupt
    if args.output_file is None:
        args.output_file = args.input+"_output"

    # What kind of model should we build?        
    if not args.crf:
        if args.awesome:
            args.model_class = EnhancedHMM
        else:
            args.model_class = HiddenMarkovModel
            if args.lexicon or args.rnn_dim:
                raise NotImplementedError("Neural HMM not implemented")
    else:
        args.model_class = ConditionalRandomField
        if args.lexicon or args.rnn_dim:
            raise NotImplementedError("Neural CRF not implemented")
    


    return args


def write_tagging(model: Union[HiddenMarkovModel, ConditionalRandomField], 
                 corpus: TaggedCorpus, 
                 output_file: Path,
                 decoder: str = "viterbi") -> None:
    """writes model predictions to file using specified decoding method"""
    #logging.info(f"Writing predictions to {output_file} using {decoder} decoder")
    
    try:
        with open(output_file, 'w') as f:
            for sentence in corpus:
                # Get predictions
                if isinstance(model, EnhancedHMM):
                    predicted = model.decode(sentence, corpus, method=decoder)
                else:
                    if decoder == "viterbi":
                        predicted = model.viterbi_tagging(sentence, corpus)
                    else:
                        predicted = model.posterior_tagging(sentence, corpus)
                
                # skipping BOS/EOS
                output_pairs = []
                for word, tag in predicted:
                    # Skip BOS/EOS tokens
                    if word in [BOS_WORD, EOS_WORD]:
                        continue
                        
                    #format OOV consistently
                    if word == OOV_WORD:
                        word = "OOV"  
                        
                    output_pairs.append(f"{word}/{tag}")
                
                print(" ".join(output_pairs), file=f)
                
    except Exception as e:
        logging.error(f"Error in write_tagging: {str(e)}")
        raise

'''

def optimize_hyperparams(model_class, train_corpus: TaggedCorpus, 
                        dev_corpus: TaggedCorpus) -> dict:
    """Find optimal hyperparameters using parallel grid search."""
    param_grid = {
        'λ': [0.01, 0.1, 0.5, 1.0, 2.0],
        'decoder': ['viterbi', 'posterior', 'hybrid'],
        'supervised_constraint': [True, False],
        'smart_smoothing': [True, False]
    }
    
    def evaluate_params(params):
        model = model_class(train_corpus.tagset, train_corpus.vocab,
                          supervised_constraint=params['supervised_constraint'],
                          smart_smoothing=params['smart_smoothing'])
        model.train(train_corpus, λ=params['λ'])
        return model_cross_entropy(model, dev_corpus)
    
    # Parallel grid search using ProcessPoolExecutor
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in product(*param_grid.values())]
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(evaluate_params, param_combinations))
    
    best_idx = np.argmin(results)
    return param_combinations[best_idx]

    '''

def main() -> None:
    args = parse_args()
    logging.root.setLevel(args.logging_level)
    logging.basicConfig(level=args.logging_level)

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device.
    # E.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration.
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    torch.set_default_device(args.device)
        
    # load or create the model
    if args.load_path:
        model = args.model_class.load(args.load_path, device=args.device)
        train_corpus = TaggedCorpus(*[Path(t) for t in args.train], 
                                  tagset=model.tagset, vocab=model.vocab)
    else:
        train_corpus = TaggedCorpus(*[Path(t) for t in args.train])
        model = args.model_class(
            train_corpus.tagset,
            train_corpus.vocab,
            unigram=args.unigram
        )

  
    eval_corpus = TaggedCorpus(Path(args.input), tagset=model.tagset, vocab=model.vocab)
    
    #  loss function
    if args.loss == 'cross_entropy':
        loss = lambda x: model_cross_entropy(x, eval_corpus)
    else:
        loss = lambda x: viterbi_error_rate(x, eval_corpus, show_cross_entropy=False)

    if train_corpus:
        if isinstance(model, ConditionalRandomField):
            model.train(
                corpus=train_corpus,
                loss=loss,
                minibatch_size=args.batch_size,
                eval_interval=args.eval_interval,
                lr=args.lr,
                reg=args.reg,
                tolerance=args.tolerance,
                max_steps=args.max_steps,
                save_path=args.save_path
            )
        else:
            model.train(
                corpus=train_corpus,
                loss=loss,
                λ=args.λ,
                tolerance=args.tolerance,
                max_steps=args.max_steps,
                save_path=args.save_path
            )
                     
    loss(model)
    
    decoder = "viterbi"  # default to viterbi
    if args.awesome:
        decoder = args.awesome_decoder
    elif hasattr(args, 'decoder'):
        decoder = args.decoder
        
    write_tagging(model, eval_corpus, Path(args.output_file), decoder=decoder)
    logging.info(f"Wrote {decoder} tagging to {args.output_file}")

if __name__ == "__main__":
    main()