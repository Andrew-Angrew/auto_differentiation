#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from circuit import Circuit
from utils import Mapping

class Optimizer:
    def __init__(self, model : Circuit, loss_function : Circuit,
             verbose=True):
        self.model = model
        self.loss_function = loss_function
        assert len(self.loss_function.output_nodes) == 1
        
        self.verbose = verbose
        self.train_err_log = []
        self.validation_err_log = []
    
    def log_errs(self, train_loss, validation_loss=None):
        self.train_err_log.append(train_loss)
        if validation_loss:
            self.validation_err_log.append(validation_loss)
        if self.verbose:
            epochs_passed = len(self.train_err_log)
            time_msg = "epoch {}:".format(epochs_passed)
            msg = "{:<10} train loss: {:<10.5}".format(time_msg, train_loss)
            if validation_loss:
                msg += "validation loss: {:<10.5}".format(validation_loss)
            print(msg)

    def fit(self, X, y,
            eval_set=None,
            n_epochs=10,
            learning_rate=0.1,
            batch_size=100,
            seed=0):
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X.shape) == 2 and len(y.shape) == 1
        assert X.shape[0] == len(y)
        assert X.shape[1] == len(self.model.input_nodes)
        assert batch_size <= len(X)
        np.random.seed(seed)
        batch_count = len(X) // batch_size
        if eval_set is not None:
            validation_X, validation_y = eval_set
            assert isinstance(validation_X, np.ndarray)
            assert isinstance(validation_y, np.ndarray)
            best_model = self.model.clone()
            best_loss, _ = self.calc_loss(validation_X, validation_y)
            self.validation_err_log.append(best_loss)

        data_permutation = np.arange(len(X))
        for epoch in range(n_epochs):
            np.random.shuffle(data_permutation)
            train_loss = 0
            validation_loss = None
            for batch_num in range(batch_count):
                batch_indexes = data_permutation[
                    batch_num * batch_size: (batch_num + 1) * batch_size
                ]
                batch_X = X[batch_indexes]
                batch_y = y[batch_indexes]
                batch_loss, batch_ders = self.calc_loss(
                    batch_X, batch_y, calc_ders=True
                )
                train_loss += batch_loss
                self.model.update_params(-learning_rate * batch_ders)
            train_loss /= batch_count
            if eval_set is not None:
                validation_loss, _ = self.calc_loss(validation_X, validation_y)
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_model = self.model.clone()
            self.log_errs(train_loss, validation_loss)
            
        if eval_set is not None:
            self.model = best_model
        return self.model

    def calc_loss(self, X, y, calc_ders=False):
        ders = Mapping()
        prediction = self.model(X.T)
        losses = self.loss_function(prediction + [y])[0]
        if calc_ders:
            self.loss_function.back_prop([np.ones_like(losses)])
            prediction_ders = self.loss_function.get_input_ders()
            prediction_ders = prediction_ders[:-1]
            self.model.back_prop(prediction_ders)
            ders = self.model.get_param_ders()
        return losses.mean(), ders.apply(np.mean)
