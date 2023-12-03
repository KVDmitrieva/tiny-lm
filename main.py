import torch
import argparse
import wandb

from torch import nn
from math import sqrt
from torch.utils.data import DataLoader

from model import MightyLanguageModel
from dataset import TextDataset
from train import train


BATCH_SIZE = 512
TRAIN_SIZE = 1.0
VOCAB_SIZE = 4000
MAX_LEN = 256

TRAIN_PATH = 'data/train.txt'
VALID_PATH = 'data/val.txt'

NUM_EPOCHS = 5

EMBED_DIM = 512
HIDDEN_DIM = 2048
N_LAYERS = 4
N_HEAD = 4
DROPOUT = 0.1

LR = 3e-4


def init_weights(model):
    for name, param in model.named_parameters():
        if len(param.data.shape) >= 2:
            nn.init.xavier_normal_(param.data, gain=sqrt(EMBED_DIM))
        else:
            nn.init.normal_(param.data, mean=0.0, std=0.01)


def main(key):
    log_wandb = key is not None
    if log_wandb:
        wandb.login(key=key)
        wandb.init(project="small_lm")

    print("Preparing data")
    train_data = TextDataset(data_file=TRAIN_PATH, vocab_size=VOCAB_SIZE, text_ratio=TRAIN_SIZE, max_length=MAX_LEN)
    val_data = TextDataset(data_file=VALID_PATH, max_length=MAX_LEN)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    print("Preparing model")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    lm_model = MightyLanguageModel(vocab_size=VOCAB_SIZE, max_len=MAX_LEN, pad_idx=train_data.pad_id, n_layers=N_LAYERS,
                                   embed_dim=EMBED_DIM, n_head=N_HEAD, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(
        device)

    lm_model.apply(init_weights)

    print(lm_model)

    optimizer = torch.optim.AdamW(lm_model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=train_data.pad_id)

    print("Start training")
    train(lm_model, optimizer, criterion, train_loader, valid_loader, num_epoch=NUM_EPOCHS, verbose=True,
          log_wandb=True)
    arch = type(lm_model).__name__
    state = {
        "arch": arch,
        "state_dict": lm_model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(state, "model.pth")
    print("Saving current model...")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-k",
        "--key",
        default=None,
        type=str,
        help="wandb key for logging",
    )
    args = args.parse_args()
    main(args.key)
