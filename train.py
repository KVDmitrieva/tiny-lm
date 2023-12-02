import torch
import wandb
from tqdm import tqdm


def train_epoch(model, optimizer, criterion, dataloader):
    history_loss = 0.0
    device = next(model.parameters()).device

    model.train()
    for padded_seq, lengths in tqdm(dataloader, desc="Train epoch", leave=False):
        smaller_pad = padded_seq[:, :lengths.max()].to(device)

        optimizer.zero_grad()
        logits = model(smaller_pad[:, :-1])
        loss = criterion(logits.transpose(1, 2), smaller_pad[:, 1:])
        loss.backward()
        optimizer.step()

        history_loss += loss.item() * len(lengths)

    return history_loss / len(dataloader)


@torch.no_grad()
def evaluate_epoch(model, criterion, dataloader):
    history_loss = 0.0
    device = next(model.parameters()).device

    model.eval()
    for padded_seq, lengths in tqdm(dataloader, desc="Val epoch", leave=False):
        smaller_pad = padded_seq[:, :lengths.max()].to(device)

        logits = model(smaller_pad[:, :-1])
        loss = criterion(logits.transpose(1, 2), smaller_pad[:, 1:])

        history_loss += loss.item() * len(lengths)

    return history_loss / len(dataloader)


def train(model, optimizer, criterion, train_loader, val_loader,
          scheduler=None, num_epoch=10, log_wandb=False, verbose=False):
    train_history, val_history = [], []
    for epoch in range(num_epoch):
        train_loss = train_epoch(model, optimizer, criterion, train_loader)
        val_loss = evaluate_epoch(model, criterion, val_loader)

        train_history.append(train_loss)
        val_history.append(val_loss)

        text_sample = model.inference(train_loader.dataset)
        text_with_prefix = model.inference(train_loader.dataset, prefix="Once upon a time,")

        if log_wandb:
            wandb.log({
                "train loss": train_loss,
                "val loss": val_loss,
                "training_samples": wandb.Html(f"<span style='color:grey;'>{text_sample}</span>"),
                "training_samples_with_prefix": wandb.Html(f"<span style='color:grey;'>{text_with_prefix}</span>")
            })

        if verbose:
            print(f"Epoch {epoch}:")
            print(f"Train loss {train_loss:.3f}")
            print(f"Validation loss {val_loss:.3f}")
            print(f"Generated sample: {text_sample}")
            print(f"Generated sample with prefix: {text_with_prefix}")

        if scheduler is not None:
            scheduler.step()
