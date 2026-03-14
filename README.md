# Modified MNIST — Two-Digit Classification with PyTorch

Simultaneous recognition of two handwritten digits from a single image using deep learning. Built with PyTorch, this project achieves **97%+ accuracy** on a modified MNIST dataset where each sample contains two vertically stacked digits.

## Sample Images

![](https://raw.githubusercontent.com/stabgan/Modified-MNIST-2-digit-classification-at-once-using-pyTorch/master/part2-twodigit/sample_images/img20000.jpg)
![](https://raw.githubusercontent.com/stabgan/Modified-MNIST-2-digit-classification-at-once-using-pyTorch/master/part2-twodigit/sample_images/img20002.jpg)
![](https://raw.githubusercontent.com/stabgan/Modified-MNIST-2-digit-classification-at-once-using-pyTorch/master/part2-twodigit/sample_images/img20004.jpg)
![](https://raw.githubusercontent.com/stabgan/Modified-MNIST-2-digit-classification-at-once-using-pyTorch/master/part2-twodigit/sample_images/img20010.jpg)

## Architecture

Two parallel neural network heads share a common input and are trained jointly using a combined cross-entropy loss.

### MLP (`mlp.py`)
- Flatten → FC(1176, 64) → FC(64, 10) (digit 1)
- Flatten → FC(1176, 64) → FC(64, 10) (digit 2)

### CNN (`conv.py`)
- Two independent convolutional pipelines, each with:
  - Conv2d(1→32) → MaxPool → Dropout(0.4)
  - Conv2d(32→32) → MaxPool → Dropout(0.2)
  - Conv2d(32→64) → MaxPool → Dropout(0.1)
  - FC(192→256) → Dropout → FC(256→10)
- Dropout regularization prevents overfitting (~97.4% with dropout, 98%+ without)

### Training
- Joint loss: `0.5 * (loss_digit1 + loss_digit2)`
- Optimizer: SGD with momentum
- Epochs: 30
- Batch size: 64
- Proper `model.eval()` + `torch.no_grad()` for validation/test inference

## Tech Stack

| Component | Technology |
|-----------|------------|
| 🧠 Framework | PyTorch |
| 🐍 Language | Python 3 |
| 📊 Data | NumPy |
| 📈 Progress | tqdm |
| 🗜️ Serialization | gzip + pickle |

## Dependencies

```
torch
numpy
tqdm
```

Install with:

```bash
pip install torch numpy tqdm
```

## How to Run

1. Place the dataset files in a `Datasets/` directory one level above the project root:
   ```
   Datasets/
   ├── train_multi_digit_mini.pkl.gz
   ├── test_multi_digit_mini.pkl.gz
   ├── train_labels_mini.txt.gz
   └── test_labels_mini.txt.gz
   ```

2. Train the CNN model:
   ```bash
   cd part2-twodigit
   python conv.py
   ```

3. Or train the MLP model:
   ```bash
   cd part2-twodigit
   python mlp.py
   ```

Set `use_mini_dataset = False` in the script to train on the full dataset (remove `_mini` suffix from filenames).

## Known Issues

- Dataset files are not included in this repo — you need to source them separately (MITx MicroMasters course material).
- The `__pycache__` directory is committed to the repo.
- `utils_multiMNIST.py` uses `_pickle` directly; consider using `pickle` for better compatibility.
- Input image dimensions are hardcoded to 42×28.
- The saved model path (`mnist_model_fully_connected.pt`) is hardcoded and overwrites on each epoch.

## License

See [LICENSE](LICENSE) for details.
