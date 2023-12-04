import torch
import argparse
import wandb

from torch import nn
from math import sqrt
from torch.utils.data import DataLoader

from model import MightyLanguageModel
from dataset import TextDataset
from train import train


BATCH_SIZE = 256
TRAIN_SIZE = 0.01
VOCAB_SIZE = 4000
MAX_LEN = 256

TRAIN_PATH = 'data/train.txt'
VALID_PATH = 'data/val.txt'

RESUME_PATH = 'model.pth'

NUM_EPOCHS = 5

EMBED_DIM = 512
HIDDEN_DIM = 1024
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


def main(prompt):
    print("Preparing data")
    val_data = TextDataset(data_file=VALID_PATH, max_length=MAX_LEN)

    print("Preparing model")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lm_model = MightyLanguageModel(vocab_size=VOCAB_SIZE, max_len=MAX_LEN, pad_idx=val_data.pad_id, n_layers=N_LAYERS,
                                   embed_dim=EMBED_DIM, n_head=N_HEAD, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)

    checkpoint = torch.load(RESUME_PATH, device)
    lm_model.load_state_dict(checkpoint["state_dict"])

    print(lm_model)

    text_sample = lm_model.inference(val_data, prompt)
    print("Generated text:")
    print(text_sample)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-p",
        "--prompt",
        default="",
        type=str,
        help="Prompt for model infernce",
    )
    args = args.parse_args()
    main(args.prompt)
