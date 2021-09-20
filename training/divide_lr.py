# -*- coding: utf-8 -*-
import numpy as np
import torch
import os.path as op

class DivideLr(object):
    """
    Launch model fine tunning if validation loss doesn't improve after
    a given patience.
    """
    def __init__(self, patience=7, verbose=False, save=False, savepath=''):
        """
        Args:
            patience (int): How long to wait after last time validation
                            loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation
                            loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.divide_lr = False
        self.val_loss_min = np.Inf
        self.save = save
        self.savepath = savepath

    def __call__(self, val_loss, model):

        if self.stop:
            self.divide_lr = False
        else:
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                if self.save:
                    self.save_checkpoint(val_loss, model)
            elif score < self.best_score:
                self.counter += 1
                print('DivideLr counter: %i out of %i' %
                      (self.counter, self.patience))
                if self.counter >= self.patience:
                    self.divide_lr = True
                    self.stop = True
            else:
                self.best_score = score
                if self.save:
                    self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased (%.6f -> %.6f). Saving model...' %
                  (self.val_loss_min, val_loss))
        torch.save(model.state_dict(), op.join(self.savepath, 'checkpoint.pt'))
        self.val_loss_min = val_loss
