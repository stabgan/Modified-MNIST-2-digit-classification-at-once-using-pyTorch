# Modified MNIST — Two-Digit Classification with PyTorch

Simultaneously classify both digits in a modified MNIST image (42×28 px) using parallel neural network branches — one branch per digit.

## What It Does

Each input image contains two vertically stacked handwritten digits. The model predicts both digits at once by splitting into two independent classification heads that share a common input but learn separate features. Training uses a joint cross-entropy loss (averaged across both heads) optimized with Adam.

Two architectures are provided:

- **MLP** (`mlp.py`) — Fully connected baseline with two parallel FC branches
- **CNN** (`conv.py`) — Convolutional network with two parallel conv+FC branches, dropout regularization, achieving ~97% accuracy

### Sample Images

<p>
  <img src="part2-twodigit/sample_images/img20000.jpg" width="56" />
  <img src="part2-twodigit/sample_images/img20002.jpg" width="56" />
  <img src="part2-twodigit/sample_images/img20004.jpg" width="56" />
  <img src="part2-twodigit/sample_images/img20010.jpg" width="56" />
  <img src="part2-twodigit/sample_images/img20014.jpg" width="56" />
</p>

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3.8+ | Language |
| 🔥 PyTorch | Deep learning framework |
| 📊 NumPy | Array operations & data loading |
| 📈 tqdm | Training progress bars |

## Project Structure

```
part2-twodigit/
├── conv.py                # CNN model — train & evaluate
├── mlp.py                 # MLP model — train & evaluate
├── train_utils.py         # Training loop, batching, accuracy
├── utils_multiMNIST.py    # Dataset loader (gzipped pickle)
└── sample_images/         # Example input images
```

## Setup

```bash
pip install -r requirements.txt
```

### Dataset

Place the dataset files in a `Datasets/` directory one level above `part2-twodigit/`:

```
Datasets/
├── train_multi_digit_mini.pkl.gz
├── test_multi_digit_mini.pkl.gz
├── train_labels_mini.txt.gz
└── test_labels_mini.txt.gz
```

### Train

```bash
cd part2-twodigit

# CNN (recommended)
python conv.py

# MLP baseline
python mlp.py
```

Training runs 30 epochs. The CNN converges to ~97.4% accuracy with dropout (>98% without).

## ⚠️ Known Issues

- Dataset files are not included in the repo — you need the MITx course materials
- `path_to_data_dir` is hardcoded to `../Datasets/`; adjust if your layout differs
- The saved model file (`mnist_model_fully_connected.pt`) is overwritten each epoch

## License

[MIT](LICENSE)
