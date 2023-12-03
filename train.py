import torch
import wandb
from tqdm import tqdm
from torch.cuda.amp import GradScaler


@torch.no_grad()
def get_grad_norm(model, norm_type=2):
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
        ),
        norm_type,
    )
    return total_norm.item()


def train_epoch(model, optimizer, criterion, dataloader, scaler, iter_accum=2):
    history_loss = 0.0
    device = next(model.parameters()).device
    amp_device = device if device == "cpu" else "cuda"

    model.train()
    for i, (padded_seq, lengths) in enumerate(tqdm(dataloader, desc="Train epoch", leave=False)):
        with torch.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=True):
            tokens = padded_seq[:, :lengths.max()].to(device)
            logits = model(tokens[:, :-1])
            loss = criterion(logits.transpose(1, 2), tokens[:, 1:]) / iter_accum

        scaler.scale(loss).backward()
        if (i + 1) % iter_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        history_loss += loss.item() * len(lengths)

    return history_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_epoch(model, criterion, dataloader, scaler):
    history_loss = 0.0
    device = next(model.parameters()).device
    amp_device = device if device == "cpu" else "cuda"

    model.eval()
    for i, (padded_seq, lengths) in enumerate(tqdm(dataloader, desc="Val epoch", leave=False)):
        with torch.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=True):
            tokens = padded_seq[:, :lengths.max()].to(device)
            logits = model(tokens[:, :-1])
            loss = criterion(logits.transpose(1, 2), tokens[:, 1:])

        history_loss += loss.item() * len(lengths)

    return history_loss / len(dataloader.dataset)


def train(model, optimizer, criterion, train_loader, val_loader,
          scheduler=None, num_epoch=10, log_wandb=False, verbose=False):
    train_history, val_history = [], []
    scaler = GradScaler(enabled=True)
    for epoch in range(num_epoch):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, scaler, scheduler)
        val_loss = evaluate_epoch(model, criterion, val_loader, scaler)

        train_history.append(train_loss)
        val_history.append(val_loss)

        text_sample = model.inference(train_loader.dataset)
        text_with_prefix = model.inference(train_loader.dataset, prefix="Once upon a time,")

        if log_wandb:
            wandb.log({
                "train loss": train_loss,
                "val loss": val_loss,
                "grad norm": get_grad_norm(model),
                "text sample": wandb.Html(f"<span style='color:grey;'>{text_sample}</span>"),
                "text sample with prefix": wandb.Html(f"<span style='color:grey;'>{text_with_prefix}</span>")
            })
            if scheduler is not None:
                wandb.log({"lr scheduler": scheduler.get_last_lr()[-1]})

        if verbose:
            print(f"Epoch {epoch}:")
            print(f"Train loss {train_loss:.3f}")
            print(f"Validation loss {val_loss:.3f}")
            print(f"Generated sample: {text_sample}")
            print()
            print(f"Generated sample with prefix: {text_with_prefix}")

        if scheduler is not None:
            scheduler.step()
