import os
import json
import argparse
import pandas as pd
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import h5py

from bytercnn_models import byte_rcnn_model, load_model
from data_loader import ByteRCNNDataLoader, NPZBatchGenerator
from utility import acc_calc

class ByteRCNNTrainer:
    def __init__(self, args):
        self.scenario_to_run = args.scenario_to_run
        self.maxlen = args.maxlen
        self.lr = args.lr
        self.embed_dim = args.embed_dim
        self.batch_size = args.batch_size
        self.kernels = args.kernels
        self.cnn_size = args.cnn_size
        self.rnn_size = args.rnn_size
        self.epochs = args.epochs
        self.output = args.output
        self.train_data_path = args.train_data_path
        self.val_data_path = args.val_data_path
        self.test_data_path = args.test_data_path
        self.olab_data_path = args.olab_data_path
        self.model_path = self.output + "/best_model.keras"
        self.model_type = args.model_type
        self._create_output_dir()

    def _create_output_dir(self):
        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def _get_model(self):
        if self.model_type == 'byte_rcnn':
            return byte_rcnn_model(self.maxlen, self.embed_dim, self.rnn_size, self.cnn_size, self.kernels, 75, self.output, self.lr)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self):
        train_generator = NPZBatchGenerator(self.train_data_path, self.batch_size)
        val_generator = NPZBatchGenerator(self.val_data_path, self.batch_size)

        model = self._get_model()

        checkpoint_filepath = os.path.join(self.output, "best_model.keras")
        history = model.fit(
            train_generator, batch_size=self.batch_size, epochs=self.epochs, validation_data=val_generator,
            callbacks=[
                keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_best_only=True)
            ]
        )

        self._plot_history(history)
        self._save_model(model)
        return model

    def _plot_history(self, history):
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.savefig(self.output + 'train_hist_acc_rcnn.png', dpi=400)
        plt.clf()

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.savefig(self.output + 'train_hist_loss_rcnn.png', dpi=400)
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
        clas_report.to_excel(self.output + f'classification_report_bytercnn{suffix}.xlsx')

    def _save_confusion_matrix(self, y_test, y_pred, labels_val, suffix=""):
        conf_matrix = confusion_matrix(y_test, y_pred, labels=labels_val)
        labels_left = np.array(labels_val).reshape(-1, 1)
        conf_matrix = np.concatenate([labels_left, conf_matrix], axis=1)
        conf_matrix = np.vstack([['-'] + labels_val, conf_matrix])

        conf_matrix_df = pd.DataFrame(conf_matrix)
        conf_matrix_df['acc'] = conf_matrix_df.apply(lambda row: acc_calc(row), axis=1)
        conf_matrix_df.to_excel(self.output + f'bytercnn_confusion_matrix{suffix}.xlsx')

    def _save_model(self, model):
        model.save(self.output + f'_bytercnn_len{self.maxlen}_sc{self.scenario_to_run}_model_save')

def main():
    parser = argparse.ArgumentParser(description="Train, Evaluate, or Test ByteRCNN Model")
    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--scenario_to_run', type=int, default=1, help='Scenario to run')
    train_parser.add_argument('--maxlen', type=int, default=4096, help='Maximum length')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--embed_dim', type=int, default=16, help='Embedding dimension')
    train_parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
    train_parser.add_argument('--kernels', type=int, nargs='+', default=[9, 27, 40, 65], help='Kernel sizes')
    train_parser.add_argument('--cnn_size', type=int, default=128, help='CNN size')
    train_parser.add_argument('--rnn_size', type=int, default=64, help='RNN size')
    train_parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    train_parser.add_argument('--output', type=str, default='mammad/', help='Output directory')
    train_parser.add_argument('--train_data_path', type=str, required=True, help='Path to training data')
    train_parser.add_argument('--val_data_path', type=str, required=True, help='Path to validation data')
    train_parser.add_argument('--test_data_path', type=str, default=None, help='Path to test data')
    train_parser.add_argument('--olab_data_path', type=str, default=None, help='Path to olab data')

    train_parser.add_argument('--model_type', type=str, default='byte_rcnn', help='Type of model to use')
    train_parser.add_argument('--model_path', type=str, default=None, help='Path to the saved model')

    eval_parser = subparsers.add_parser('evaluate')
    eval_parser.add_argument('--scenario_to_run', type=int, default=1, help='Scenario to run')
    eval_parser.add_argument('--maxlen', type=int, default=4096, help='Maximum length')
    eval_parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
    eval_parser.add_argument('--output', type=str, default='mammad/', help='Output directory')
    eval_parser.add_argument('--test_data_path', type=str, required=True, help='Path to test data')
    eval_parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--scenario_to_run', type=int, default=1, help='Scenario to run')
    test_parser.add_argument('--maxlen', type=int, default=4096, help='Maximum length')
    test_parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
    test_parser.add_argument('--output', type=str, default='mammad/', help='Output directory')
    test_parser.add_argument('--olab_data_path', type=str, required=True, help='Path to olab data')
    test_parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')

    args = parser.parse_args()

    trainer = ByteRCNNTrainer(args)

    if args.command == 'train':
        trainer.train()
    elif args.command == 'evaluate':
        trainer.evaluate()
    elif args.command == 'test':
        trainer.test()

if __name__ == "__main__":
    main()
