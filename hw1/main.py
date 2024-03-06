# Numerical Operations
import math
import numpy as np

# Reading/Writing data
import pandas as pd
import os
import csv

# For process Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter

from config import config, device
from utility import *
from dataset import COVID19Dataset
from model import My_Model

def select_feat(train_data, valid_data, test_data, select_all=True):
    """Select useful features to perform regression"""
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = list(range(35,raw_x_train.shape[1])) # TODO: Select suitable feature columns.

    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid

def trainer(train_loader, valid_loader, model):
    criterion = nn.MSELoss(reduction="mean")

    # Define your optimization algorithm.
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.7)
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter()

    if not os.path.isdir("./models"):
        os.mkdir("./models")

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0,0 # step-update counts

    # Train
    for epoch in range(n_epochs):
        model.train()
        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        # get one batch, and update
        for x,y in train_pbar:
            optimizer.zero_grad()
            x,y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar. 前后缀
            train_pbar.set_description(f"Epoch [{epoch+1}/{n_epochs}]")
            train_pbar.set_postfix({"loss": loss.detach().item()}) # dictionary

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar("Loss/Train", mean_train_loss, step)

        # once update, valid
        model.eval()
        loss_record = []
        for x,y in valid_loader:
            x,y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f"Epoch [{epoch+1}/{n_epochs}]: Train Loss: {mean_train_loss:.4f}, Valid Loss: {mean_valid_loss:.4f}")
        writer.add_scalar("Loss/Valid", mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config["save_path"])
            print("Saving best model with loss: {:.3f}...".format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return

def save_pred(preds, file):
    """Save predictions to specified file"""
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i,p in enumerate(preds):
            writer.writerow([i,p])

if __name__ == "__main__":
    # set seed for reproducibility
    same_seed(config["seed"])

    # select_feat function need numpy format data, so need pd.read_csv().values
    train_data, test_data = pd.read_csv("./covid_train.csv").values, pd.read_csv("./covid_test.csv").values
    train_data, valid_data = train_valid_split(train_data, config["valid_ratio"], config["seed"])

    # Print out the data size.
    print(f"""train_data size: {train_data.shape} 
    valid_data size: {valid_data.shape} 
    test_data size: {test_data.shape}""")

    x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data,select_all=config["select_all"])

    # Print out the number of features.
    print(f'number of features: {x_train.shape[1]}')

    train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), COVID19Dataset(x_valid, y_valid), COVID19Dataset(x_test)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True)

    model = My_Model(input_dim=x_train.shape[1]).to(device)  # put your model and data on the same computation device.
    trainer(train_loader, valid_loader, model)

    # Testing
    model = My_Model(input_dim=x_train.shape[1]).to(device)
    model.load_state_dict(torch.load(config["save_path"]))
    preds = predict(test_loader, model, device)
    save_pred(preds, 'pred.csv')

