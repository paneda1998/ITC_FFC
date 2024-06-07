import os
import json
import argparse
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pickle

from bytercnn_models import byte_rcnn_model, load_model
from data_loader import ByteRCNNDataLoader, NPZBatchGenerator
from sklearn_models import decision_tree_model, random_forest_model, simple_cnn_model
from utility import acc_calc
from config_parser import parse_config


class Trainer:
    def __init__(self, config_path):
        config = parse_config(config_path)
        self.config = config
        self.model_type = config['model_type']
        self._load_common_config()
        self._load_model_specific_config()
        self._create_output_dir()

    def _load_common_config(self):
        config = self.config
        self.scenario_to_run = config['scenario_to_run']
        self.maxlen = config['maxlen']
        self.batch_size = config['batch_size']
        self.output = config['output']
        self.train_data_path = config['train_data_path']
        self.val_data_path = config['val_data_path']
        self.test_data_path = config['test_data_path']
        self.olab_data_path = config['olab_data_path']
        self.checkpoint_dir = config.get('checkpoint_dir', self.output)

    def _load_model_specific_config(self):
        config = self.config
        if self.model_type == 'byte_rcnn':
            self.lr = config['lr']
            self.embed_dim = config['embed_dim']
            self.kernels = config['kernels']
            self.cnn_size = config['cnn_size']
            self.rnn_size = config['rnn_size']
            self.epochs = config['epochs']
        elif self.model_type == 'decision_tree':
            self.criterion = config['criterion']
            self.splitter = config['splitter']
            self.max_depth = config['max_depth']
            self.min_samples_split = config['min_samples_split']
            self.min_samples_leaf = config['min_samples_leaf']
        elif self.model_type == 'random_forest':
            self.n_estimators = config['n_estimators']
            self.criterion = config['criterion']
            self.max_depth = config['max_depth']

    def _create_output_dir(self):
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def _load_data(self):
        # Load and prepare data based on the configuration
        if self.model_type == 'byte_rcnn':
            self.data_loader = ByteRCNNDataLoader(self.train_data_path, self.val_data_path, self.test_data_path,
                                                  self.olab_data_path, self.batch_size, self.maxlen)
            self.train_generator, self.val_generator, self.test_generator = self.data_loader.load_data()
        else:
            self.train_data = pd.read_csv(self.train_data_path)
            self.val_data = pd.read_csv(self.val_data_path)
            self.test_data = pd.read_csv(self.test_data_path)

    def train(self):
        self._load_data()
        if self.model_type == 'byte_rcnn':
            self._train_byte_rcnn()
        elif self.model_type == 'decision_tree':
            self._train_decision_tree()
        elif self.model_type == 'random_forest':
            self._train_random_forest()

    def _train_byte_rcnn(self):
        model = byte_rcnn_model(self.maxlen, self.embed_dim, self.kernels, self.cnn_size, self.rnn_size)
        checkpoint_path = os.path.join(self.checkpoint_dir, 'byte_rcnn_checkpoint.h5')

        # Load weights if checkpoint exists
        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)

        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True,
                                              verbose=1)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(self.train_generator, epochs=self.epochs, validation_data=self.val_generator,
                  callbacks=[checkpoint_callback])
        model.save(os.path.join(self.output, 'byte_rcnn_model.h5'))

    def _train_decision_tree(self):
        model = decision_tree_model(self.criterion, self.splitter, self.max_depth, self.min_samples_split,
                                    self.min_samples_leaf)
        X_train = self.train_data.drop('label', axis=1)
        y_train = self.train_data['label']
        model.fit(X_train, y_train)
        with open(os.path.join(self.output, 'decision_tree_model.pkl'), 'wb') as f:
            pickle.dump(model, f)

    def _train_random_forest(self):
        model = random_forest_model(self.n_estimators, self.criterion, self.max_depth)
        X_train = self.train_data.drop('label', axis=1)
        y_train = self.train_data['label']
        model.fit(X_train, y_train)
        with open(os.path.join(self.output, 'random_forest_model.pkl'), 'wb') as f:
            pickle.dump(model, f)

    def test(self):
        self._load_data()
        if self.model_type == 'byte_rcnn':
            self._test_byte_rcnn()
        elif self.model_type == 'decision_tree':
            self._test_decision_tree()
        elif self.model_type == 'random_forest':
            self._test_random_forest()

    def _test_byte_rcnn(self):
        model = load_model(os.path.join(self.output, 'byte_rcnn_model.h5'))
        results = model.evaluate(self.test_generator)
        print(f'Test Loss: {results[0]}, Test Accuracy: {results[1]}')

        y_pred = model.predict(self.test_generator)
        y_true = np.concatenate([y for x, y in self.test_generator], axis=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)

        self._generate_classification_report(y_true_classes, y_pred_classes)
        self._generate_confusion_matrix(y_true_classes, y_pred_classes)

    def _test_decision_tree(self):
        with open(os.path.join(self.output, 'decision_tree_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        X_test = self.test_data.drop('label', axis=1)
        y_test = self.test_data['label']
        y_pred = model.predict(X_test)

        self._generate_classification_report(y_test, y_pred)
        self._generate_confusion_matrix(y_test, y_pred)

    def _test_random_forest(self):
        with open(os.path.join(self.output, 'random_forest_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        X_test = self.test_data.drop('label', axis=1)
        y_test = self.test_data['label']
        y_pred = model.predict(X_test)

        self._generate_classification_report(y_test, y_pred)
        self._generate_confusion_matrix(y_test, y_pred)

    def cross_validate(self):
        self._load_data()
        if self.model_type == 'decision_tree':
            self._cross_validate_decision_tree()
        elif self.model_type == 'random_forest':
            self._cross_validate_random_forest()

    def _cross_validate_decision_tree(self):
        model = decision_tree_model(self.criterion, self.splitter, self.max_depth, self.min_samples_split,
                                    self.min_samples_leaf)
        X = self.train_data.drop('label', axis=1)
        y = self.train_data['label']
        scores = cross_val_score(model, X, y, cv=5)
        print(f'Cross-Validation Scores: {scores}')
        print(f'Mean CV Score: {np.mean(scores)}')

    def _cross_validate_random_forest(self):
        model = random_forest_model(self.n_estimators, self.criterion, self.max_depth)
        X = self.train_data.drop('label', axis=1)
        y = self.train_data['label']
        scores = cross_val_score(model, X, y, cv=5)
        print(f'Cross-Validation Scores: {scores}')
        print(f'Mean CV Score: {np.mean(scores)}')

    def _generate_classification_report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(self.output, 'classification_report.csv')
        report_df.to_csv(report_path, index=True)
        print(f'Classification report saved to {report_path}')

    def _generate_confusion_matrix(self, y_true, y_pred):
        matrix = confusion_matrix(y_true, y_pred)
        matrix_path = os.path.join(self.output, 'confusion_matrix.csv')
        np.savetxt(matrix_path, matrix, delimiter=",")
        print(f'Confusion matrix saved to {matrix_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train, test, or cross-validate models.")
    parser.add_argument('method', choices=['train', 'test', 'cross_validate'], help="Method to execute.")
    parser.add_argument('config', help="Path to the configuration file.")

    args = parser.parse_args()

    trainer = Trainer(args.config)

    if args.method == 'train':
        trainer.train()
    elif args.method == 'test':
        trainer.test()
    elif args.method == 'cross_validate':
        trainer.cross_validate()
