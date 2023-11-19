import torch

from torch import nn
from math import sqrt
from torch.utils.data import DataLoader

from model import MightyLanguageModel
from dataset import TextDataset
from train import train


BATCH_SIZE = 64
TRAIN_SIZE = 0.1
VOCAB_SIZE = 16000
MAX_LEN = 1024

TRAIN_PATH = '/kaggle/input/tiny-stories-ds/TinyStoriesV3-GPT4-train.txt'
VALID_PATH = '/kaggle/input/tiny-stories-ds/TinyStoriesV3-GPT4-valid.txt'

NUM_EPOCHS = 5

EMBED_DIM = 64
HIDDEN_DIM = 128
N_LAYERS = 1
N_HEAD = 4
DROPOUT = 0.1

LR = 4e-3
GAMMA = 0.1
MILESTONE = [3, 7, 10, 13]


def init_weights(model):
    for name, param in model.named_parameters():
        if len(param.data.shape) >= 2:
            nn.init.xavier_normal_(param.data, gain=sqrt(EMBED_DIM))
        else:
            nn.init.normal_(param.data, mean=0.0, std=0.1)


def main():
    print("Preparing data")
    train_data = TextDataset(data_file=TRAIN_PATH, vocab_size=VOCAB_SIZE, text_ratio=TRAIN_SIZE, max_length=MAX_LEN)
    val_data = TextDataset(data_file=VALID_PATH, max_length=MAX_LEN)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    print("Preparing model")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    lm_model = MightyLanguageModel(vocab_size=VOCAB_SIZE, max_len=MAX_LEN, pad_idx=train_data.pad_id, n_layers=N_LAYERS,
                                   embed_dim=EMBED_DIM, n_head=N_HEAD, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)

    lm_model.apply(init_weights)

    optimizer = torch.optim.Adam(lm_model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=NUM_EPOCHS)
    criterion = torch.nn.CrossEntropyLoss()

    print("Start training")
    train(lm_model, optimizer, criterion, train_loader, valid_loader, scheduler, NUM_EPOCHS, verbose=True)

    arch = type(lm_model).__name__
    state = {
        "arch": arch,
        "state_dict": lm_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict()
    }

    torch.save(state, "model.pth")
    print("Saving current model...")


if __name__ == '__main__':
    main()
