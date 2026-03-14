import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions



class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        self.flatten = Flatten
        self.input_dimension = input_dimension
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(192, 256)
        self.fc2 = nn.Linear(256, 10)

        self.conv4 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc3 = nn.Linear(192, 256)
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):

        xf = x.clone()
        x = F.relu(F.max_pool2d(self.conv1(xf), 2))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.1, training=self.training)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x1 = self.fc2(x)

        x = F.relu(F.max_pool2d(self.conv4(xf), 2))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(F.max_pool2d(self.conv5(x), 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(F.max_pool2d(self.conv6(x), 2))
        x = F.dropout(x, p=0.1, training=self.training)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x2 = self.fc2(x)
        

        return x1,x2

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension) # TODO add proper layers to CNN class above

    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
