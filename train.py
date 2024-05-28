import os
import json
import argparse
import pandas as pd
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import h5py

from bytercnn_models import byte_rcnn_model, load_model
from data_loader import ByteRCNNDataLoader, NPZBatchGenerator
from sklearn_models import decision_tree_model, random_forest_model, simple_cnn_model
from utility import acc_calc
from config_parser import parse_config

class Trainer:
    def __init__(self, args):
        config = parse_config(args.config)
        self.model_type = config['model_type']

        # Load common configurations
        self.scenario_to_run = config['scenario_to_run']
        self.maxlen = config['maxlen']
        self.batch_size = config['batch_size']
        self.output = config['output']
        self.train_data_path = config['train_data_path']
        self.val_data_path = config['val_data_path']
        self.test_data_path = config['test_data_path']
        self.olab_data_path = config['olab_data_path']
        self.model_type = config['model_type']
        self._create_output_dir()

        # Load model-specific configurations
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
            self.min_samples_split = config['min_samples_split']
            self.min_samples_leaf = config['min_samples_leaf']
        elif self.model_type == 'simple_cnn':
            self.lr = config['lr']
            self.embed_dim = config['embed_dim']
            self.epochs = config['epochs']
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _create_output_dir(self):
        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def _get_model(self):
        if self.model_type == 'byte_rcnn':
            return byte_rcnn_model(self.maxlen, self.embed_dim, self.rnn_size, self.cnn_size, self.kernels, 75, self.output, self.lr)
        elif self.model_type == 'decision_tree':
            return decision_tree_model(
                criterion=self.criterion,
                splitter=self.splitter,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
        elif self.model_type == 'random_forest':
            return random_forest_model(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
        elif self.model_type == 'simple_cnn':
            return simple_cnn_model((self.maxlen, self.embed_dim, 1), 75)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _get_model(self):
        if self.model_type == 'byte_rcnn':
            return byte_rcnn_model(self.maxlen, self.embed_dim, self.rnn_size, self.cnn_size, self.kernels, 75, self.output, self.lr)
        elif self.model_type == 'decision_tree':
            return decision_tree_model(
                criterion=self.args.criterion,
                splitter=self.args.splitter,
                max_depth=self.args.max_depth,
                min_samples_split=self.args.min_samples_split,
                min_samples_leaf=self.args.min_samples_leaf
            )
        elif self.model_type == 'random_forest':
            return random_forest_model(
                n_estimators=self.args.n_estimators,
                criterion=self.args.criterion,
                max_depth=self.args.max_depth,
                min_samples_split=self.args.min_samples_split,
                min_samples_leaf=self.args.min_samples_leaf
            )
        elif self.model_type == 'simple_cnn':
            return simple_cnn_model((self.maxlen, self.embed_dim, 1), 75)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self):
        train_generator = NPZBatchGenerator(self.train_data_path, self.batch_size)
        val_generator = NPZBatchGenerator(self.val_data_path, self.batch_size)

        model = self._get_model()

        if self.model_type in ['byte_rcnn', 'simple_cnn']:
            checkpoint_filepath = os.path.join(self.output, "best_model.keras")
            history = model.fit(
                train_generator, batch_size=self.batch_size, epochs=self.epochs, validation_data=val_generator,
                callbacks=[
                    keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_best_only=True)
                ]
            )
            self._plot_history(history)
        else:
            x_train, y_train = train_generator[0]  # Assuming data fits into memory
            x_val, y_val = val_generator[0]

            model = self._get_model()
            model.fit(x_train, y_train)

            # Evaluate on validation data
            y_pred = model.predict(x_val)
            print(classification_report(y_val, y_pred))

        self._save_model(model)
        return model

    def _plot_history(self, history):
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.savefig(self.output + 'train_hist_acc.png', dpi=400)
        plt.clf()

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.savefig(self.output + 'train_hist_loss.png', dpi=400)
        plt.clf()

    def evaluate(self):
        model = load_model(self.model_path)
        x_test, y_test, labels_val = ByteRCNNDataLoader.load_npz_data(self.scenario_to_run, '4k' if self.maxlen == 4096 else str(self.maxlen), 'test')
        self._evaluate_and_report(model, x_test, y_test, labels_val, suffix="")

    def test(self):
        model = load_model(self.model_path)
        x_test_olab = np.load(self.olab_data_path + 'combined_data.npy')
        y_test_olab = np.load(self.olab_data_path + 'labels.npy')
        self._evaluate_and_report(model, x_test_olab, y_test_olab, suffix="_olab")

    def _evaluate_and_report(self, model, x_test, y_test, labels_val, suffix=""):
        results = model.evaluate(x_test, y_test, batch_size=self.batch_size)
        print(f"test loss, test acc{suffix}:", results)

        y_pred = model.predict(x_test, batch_size=self.batch_size)
        y_pred = np.argmax(y_pred, axis=-1)

        y_pred = [labels_val[a] for a in y_pred]
        y_test = [labels_val[a] for a in y_test]

        self._save_classification_report(y_test, y_pred, labels_val, suffix)
        self._save_confusion_matrix(y_test, y_pred, labels_val, suffix)

    def _save_classification_report(self, y_test, y_pred, labels_val, suffix=""):
        report = classification_report(y_test, y_pred, target_names=labels_val, output_dict=True)
        clas_report = pd.DataFrame(report).transpose()
        clas_report.to_excel(self.output + f'classification_report{suffix}.xlsx')

    def _save_confusion_matrix(self, y_test, y_pred, labels_val, suffix=""):
        conf_matrix = confusion_matrix(y_test, y_pred, labels=labels_val)
        labels_left = np.array(labels_val).reshape(-1, 1)
        conf_matrix = np.concatenate([labels_left, conf_matrix], axis=1)
        conf_matrix = np.vstack([['-'] + labels_val, conf_matrix])

        conf_matrix_df = pd.DataFrame(conf_matrix)
        conf_matrix_df['acc'] = conf_matrix_df.apply(lambda row: acc_calc(row), axis=1)
        conf_matrix_df.to_excel(self.output + f'confusion_matrix{suffix}.xlsx')

    def _save_model(self, model):
        model.save(self.output + f'_model_len{self.maxlen}_sc{self.scenario_to_run}_model_save')

def main():
    parser = argparse.ArgumentParser(description="Train, Evaluate, or Test ByteRCNN Model")
    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')

    eval_parser = subparsers.add_parser('evaluate')
    eval_parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')

    args = parser.parse_args()

    # Load parameters from config file
    config = parse_config(args.config)

    trainer = Trainer(args)

    if args.command == 'train':
        trainer.train()
    elif args.command == 'evaluate':
        trainer.evaluate()
    elif args.command == 'test':
        trainer.test()

if __name__ == "__main__":
    main()
