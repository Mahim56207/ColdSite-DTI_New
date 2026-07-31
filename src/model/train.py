"""
Track B (124AD0015) — training loop.

Standard PyTorch training loop: batch, forward, loss, backward, step.
Fill in the DataLoader / dataset class once real split files from
Track A (data/splits/) are ready. See docs/02_GUIDE_124AD0015.md Step 4-5.
"""
import torch
import torch.nn as nn

from src.model.coldsite_dti import ColdSiteDTI


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for drug_batch, protein_batch, label_batch in dataloader:
        drug_batch = drug_batch.to(device)
        protein_batch = protein_batch.to(device)
        label_batch = label_batch.to(device)

        optimizer.zero_grad()
        pred, _attn = model(drug_batch, protein_batch)
        loss = loss_fn(pred.squeeze(-1), label_batch.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * drug_batch.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for drug_batch, protein_batch, label_batch in dataloader:
        drug_batch = drug_batch.to(device)
        protein_batch = protein_batch.to(device)
        label_batch = label_batch.to(device)
        pred, _attn = model(drug_batch, protein_batch)
        loss = loss_fn(pred.squeeze(-1), label_batch.float())
        total_loss += loss.item() * drug_batch.size(0)
    return total_loss / len(dataloader.dataset)


def run_training(drug_vocab_size, protein_vocab_size, train_loader, val_loader,
                  n_epochs=30, lr=1e-3, checkpoint_path="results/coldsite_dti_best.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    model = ColdSiteDTI(drug_vocab_size, protein_vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()   # switch to nn.BCEWithLogitsLoss() if framing as classification

    best_val_loss = float("inf")
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}/{n_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> saved new best checkpoint to {checkpoint_path}")

    return model


if __name__ == "__main__":
    print("Run this after building a Dataset/DataLoader around data/splits/*.csv")
    print("See docs/02_GUIDE_124AD0015.md for the full walkthrough.")
