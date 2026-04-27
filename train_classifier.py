
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from src.classifier import SmallCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', type=str, default='models/quality_classifier.pt')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dir = Path(args.data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'

    tfm_train = T.Compose([
        T.Resize((160, 160)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tfm_val = T.Compose([
        T.Resize((160, 160)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds_train = ImageFolder(str(train_dir), transform=tfm_train)
    ds_val = ImageFolder(str(val_dir), transform=tfm_val)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SmallCNN(num_classes=len(ds_train.classes)).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    best_val = 1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        correct = 0

        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * xb.size(0)
            total_n += xb.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())

        train_loss = total_loss / max(1, total_n)
        train_acc = correct / max(1, total_n)

        model.eval()
        vloss = 0.0
        vn = 0
        vcorrect = 0

        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                vloss += float(loss.item()) * xb.size(0)
                vn += xb.size(0)
                pred = torch.argmax(logits, dim=1)
                vcorrect += int((pred == yb).sum().item())

        val_loss = vloss / max(1, vn)
        val_acc = vcorrect / max(1, vn)

        print('Epoch')
        print(epoch)
        print('Train loss')
        print(round(train_loss, 4))
        print('Train acc')
        print(round(train_acc, 4))
        print('Val loss')
        print(round(val_loss, 4))
        print('Val acc')
        print(round(val_acc, 4))
        print('---')

        if val_loss < best_val:
            best_val = val_loss
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)

    print('Saved best model to')
    print(args.out)


if __name__ == '__main__':
    main()
