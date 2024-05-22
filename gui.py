import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess

class ByteRCNNGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ByteRCNN Model GUI")

        self.create_widgets()

    def create_widgets(self):
        # Model Type
        tk.Label(self.root, text="Model Type:").grid(row=0, column=0, sticky=tk.W)
        self.model_type = tk.StringVar(value="byte_rcnn")
        tk.Entry(self.root, textvariable=self.model_type).grid(row=0, column=1, sticky=tk.EW)

        # Scenario to Run
        tk.Label(self.root, text="Scenario to Run:").grid(row=1, column=0, sticky=tk.W)
        self.scenario_to_run = tk.IntVar(value=1)
        tk.Entry(self.root, textvariable=self.scenario_to_run).grid(row=1, column=1, sticky=tk.EW)

        # Max Length
        tk.Label(self.root, text="Max Length:").grid(row=2, column=0, sticky=tk.W)
        self.maxlen = tk.IntVar(value=4096)
        tk.Entry(self.root, textvariable=self.maxlen).grid(row=2, column=1, sticky=tk.EW)

        # Learning Rate
        tk.Label(self.root, text="Learning Rate:").grid(row=3, column=0, sticky=tk.W)
        self.lr = tk.DoubleVar(value=0.001)
        tk.Entry(self.root, textvariable=self.lr).grid(row=3, column=1, sticky=tk.EW)

        # Embedding Dimension
        tk.Label(self.root, text="Embedding Dimension:").grid(row=4, column=0, sticky=tk.W)
        self.embed_dim = tk.IntVar(value=16)
        tk.Entry(self.root, textvariable=self.embed_dim).grid(row=4, column=1, sticky=tk.EW)

        # Batch Size
        tk.Label(self.root, text="Batch Size:").grid(row=5, column=0, sticky=tk.W)
        self.batch_size = tk.IntVar(value=200)
        tk.Entry(self.root, textvariable=self.batch_size).grid(row=5, column=1, sticky=tk.EW)

        # Kernels
        tk.Label(self.root, text="Kernels:").grid(row=6, column=0, sticky=tk.W)
        self.kernels = tk.StringVar(value="9 27 40 65")
        tk.Entry(self.root, textvariable=self.kernels).grid(row=6, column=1, sticky=tk.EW)

        # CNN Size
        tk.Label(self.root, text="CNN Size:").grid(row=7, column=0, sticky=tk.W)
        self.cnn_size = tk.IntVar(value=128)
        tk.Entry(self.root, textvariable=self.cnn_size).grid(row=7, column=1, sticky=tk.EW)

        # RNN Size
        tk.Label(self.root, text="RNN Size:").grid(row=8, column=0, sticky=tk.W)
        self.rnn_size = tk.IntVar(value=64)
        tk.Entry(self.root, textvariable=self.rnn_size).grid(row=8, column=1, sticky=tk.EW)

        # Epochs
        tk.Label(self.root, text="Epochs:").grid(row=9, column=0, sticky=tk.W)
        self.epochs = tk.IntVar(value=30)
        tk.Entry(self.root, textvariable=self.epochs).grid(row=9, column=1, sticky=tk.EW)

        # Output Directory
        tk.Label(self.root, text="Output Directory:").grid(row=10, column=0, sticky=tk.W)
        self.output = tk.StringVar(value="mammad/")
        tk.Entry(self.root, textvariable=self.output).grid(row=10, column=1, sticky=tk.EW)

        # Train Data Path
        tk.Label(self.root, text="Train Data Path:").grid(row=11, column=0, sticky=tk.W)
        self.train_data_path = tk.StringVar()
        tk.Entry(self.root, textvariable=self.train_data_path).grid(row=11, column=1, sticky=tk.EW)
        tk.Button(self.root, text="Browse", command=self.browse_train_data).grid(row=11, column=2, sticky=tk.EW)

        # Validation Data Path
        tk.Label(self.root, text="Validation Data Path:").grid(row=12, column=0, sticky=tk.W)
        self.val_data_path = tk.StringVar()
        tk.Entry(self.root, textvariable=self.val_data_path).grid(row=12, column=1, sticky=tk.EW)
        tk.Button(self.root, text="Browse", command=self.browse_val_data).grid(row=12, column=2, sticky=tk.EW)

        # Test Data Path
        tk.Label(self.root, text="Test Data Path:").grid(row=13, column=0, sticky=tk.W)
        self.test_data_path = tk.StringVar()
        tk.Entry(self.root, textvariable=self.test_data_path).grid(row=13, column=1, sticky=tk.EW)
        tk.Button(self.root, text="Browse", command=self.browse_test_data).grid(row=13, column=2, sticky=tk.EW)

        # OLAB Data Path
        tk.Label(self.root, text="OLAB Data Path:").grid(row=14, column=0, sticky=tk.W)
        self.olab_data_path = tk.StringVar()
        tk.Entry(self.root, textvariable=self.olab_data_path).grid(row=14, column=1, sticky=tk.EW)
        tk.Button(self.root, text="Browse", command=self.browse_olab_data).grid(row=14, column=2, sticky=tk.EW)

        # Model Path
        tk.Label(self.root, text="Model Path:").grid(row=15, column=0, sticky=tk.W)
        self.model_path = tk.StringVar()
        tk.Entry(self.root, textvariable=self.model_path).grid(row=15, column=1, sticky=tk.EW)
        tk.Button(self.root, text="Browse", command=self.browse_model).grid(row=15, column=2, sticky=tk.EW)

        # Buttons for actions
        tk.Button(self.root, text="Train", command=self.train).grid(row=16, column=0, sticky=tk.EW)
        tk.Button(self.root, text="Evaluate", command=self.evaluate).grid(row=16, column=1, sticky=tk.EW)
        tk.Button(self.root, text="Test", command=self.test).grid(row=16, column=2, sticky=tk.EW)

    def browse_train_data(self):
        self.train_data_path.set(filedialog.askopenfilename())

    def browse_val_data(self):
        self.val_data_path.set(filedialog.askopenfilename())

    def browse_test_data(self):
        self.test_data_path.set(filedialog.askopenfilename())

    def browse_olab_data(self):
        self.olab_data_path.set(filedialog.askdirectory())

    def browse_model(self):
        self.model_path.set(filedialog.askopenfilename())

    def train(self):
        command = [
            'python', 'train.py', 'train',
            '--model_type', self.model_type.get(),
            '--scenario_to_run', str(self.scenario_to_run.get()),
            '--maxlen', str(self.maxlen.get()),
            '--lr', str(self.lr.get()),
            '--embed_dim', str(self.embed_dim.get()),
            '--batch_size', str(self.batch_size.get()),
            '--kernels', *self.kernels.get().split(),
            '--cnn_size', str(self.cnn_size.get()),
            '--rnn_size', str(self.rnn_size.get()),
            '--epochs', str(self.epochs.get()),
            '--output', self.output.get(),
            '--train_data_path', self.train_data_path.get(),
            '--val_data_path', self.val_data_path.get()
        ]
        self.run_command(command)

    def evaluate(self):
        command = [
            'python', 'train.py', 'evaluate',
            '--model_type', self.model_type.get(),
            '--scenario_to_run', str(self.scenario_to_run.get()),
            '--maxlen', str(self.maxlen.get()),
            '--batch_size', str(self.batch_size.get()),
            '--output', self.output.get(),
            '--test_data_path', self.test_data_path.get(),
            '--model_path', self.model_path.get()
        ]
        self.run_command(command)

    def test(self):
        command = [
            'python', 'train.py', 'test',
            '--model_type', self.model_type.get(),
            '--scenario_to_run', str(self.scenario_to_run.get()),
            '--maxlen', str(self.maxlen.get()),
            '--batch_size', str(self.batch_size.get()),
            '--output', self.output.get(),
            '--olab_data_path', self.olab_data_path.get(),
            '--model_path', self.model_path.get()
        ]
        self.run_command(command)

    def run_command(self, command):
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            messagebox.showinfo("Success", result.stdout)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", e.stderr)

if __name__ == "__main__":
    root = tk.Tk()
    app = ByteRCNNGUI(root)
    root.mainloop()
