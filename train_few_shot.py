import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import Sampler
from torchvision import datasets, models, transforms
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EpisodicBatchSampler(Sampler):
    """
    Samples batches for episodic training (prototypical network)
    Each batch has data for the 13 classes (pieces + empty square), and for each class, k_shot + q_query samples
    """

    def __init__(self, labels, n_way, k_shot, q_query, n_episodes):
        super().__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_episodes = n_episodes

        self.classes = list(set(labels))
        self.indices_per_class = {c: [] for c in self.classes}
        for i, label in enumerate(labels):
            self.indices_per_class[label].append(i)

        # Check if we have enough samples per class (enough for support + query set)
        for c in self.classes:
            assert len(self.indices_per_class[c]) >= k_shot + q_query, \
                f"Class {c} has only {len(self.indices_per_class[c])} samples, but need {k_shot + q_query}"

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            batch = []
            # Sample for each class
            classes = random.sample(self.classes, self.n_way)
            for c in classes:
                # Sample k_shot + q_query images from this class
                l = random.sample(self.indices_per_class[c], self.k_shot + self.q_query)
                batch.extend(l)
            yield batch


def get_euclidean_distances(x, y):
    """
    Computes pairwise Euclidean distances between x and y
    """

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception("x and y feature dimensions must match")

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


class PrototypicalNet(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super().__init__()
        
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.out_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        # by default, we use the less complex ResNet18
        elif backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.out_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported : {backbone_name}")

    def forward(self, x):
        return self.backbone(x)

    def compute_loss_and_acc(self, embeddings, n_way, k_shot, q_query):
        """
        Embeddings shape: (n_way * (k_shot + q_query), out_dim)
        """

        device = embeddings.device
        
        # Reshape to (n_way, k_shot + q_query, out_dim)
        embeddings = embeddings.view(n_way, k_shot + q_query, -1)
        
        # Split into support and query sets
        support = embeddings[:, :k_shot, :]
        query = embeddings[:, k_shot:, :]
        
        # Calculate prototypes (mean of the embeddings from the support set for each class)
        prototypes = support.mean(dim=1)     # (n_way, out_dim)
        
        # Reshape query for distance calculation
        query = query.contiguous().view(n_way * q_query, -1) # (n_way * q_query, out_dim)
        
        # Compute distances
        dists = get_euclidean_distances(query, prototypes) # (n_way * q_query, n_way)
        
        # Log probabilities
        log_p_y = F.log_softmax(-dists, dim=1) # (n_way * q_query, n_way)
        
        # Create targets
        target_inds = torch.arange(n_way).view(n_way, 1).expand(n_way, q_query).long() # (n_way, q_query)
        target_inds = target_inds.contiguous().view(-1).to(device) # (n_way * q_query)
        
        # Compute loss
        loss = F.nll_loss(log_p_y, target_inds)
        
        # Compute accuracy
        _, y_hat = log_p_y.max(1)
        acc = torch.eq(y_hat, target_inds).float().mean()
        
        return loss, acc


def train_epoch(model, dataloader, optimizer, n_way, k_shot, q_query, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        
        optimizer.zero_grad()
        embeddings = model(images)
        loss, acc = model.compute_loss_and_acc(embeddings, n_way, k_shot, q_query)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc.item()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc.item():.4f}"})
        
    return total_loss / len(dataloader), total_acc / len(dataloader)


@torch.no_grad()
def eval_epoch(model, dataloader, n_way, k_shot, q_query, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    
    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)
        embeddings = model(images)
        loss, acc = model.compute_loss_and_acc(embeddings, n_way, k_shot, q_query)
        
        total_loss += loss.item()
        total_acc += acc.item()
        
    return total_loss / len(dataloader), total_acc / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train Prototypical Network for Chess Pieces")
    parser.add_argument("--data-dir", default="Dataset/piece_crops", help="Path to cropped dataset")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--n-way", type=int, default=13, help="Number of classes per episode")
    parser.add_argument("--k-shot", type=int, default=5, help="Number of support examples per class")
    parser.add_argument("--q-query", type=int, default=5, help="Number of query examples per class")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes per epoch")
    parser.add_argument("--val-episodes", type=int, default=50, help="Number of validation episodes")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--out-dir", default="checkpoints", help="Output directory for weights")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset preparation (we modify the images so that the model sees diff. angles, lighting...)
    # Train dataset
    train_transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Validation dataset
    val_transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(args.data_dir)
    print(f"Found {len(full_dataset)} total images belonging to {len(full_dataset.classes)} classes.")
    
    # Split for train/val (80/20)
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    random.shuffle(indices)
    split = int(np.floor(0.2 * num_samples))
    train_idx, val_idx = indices[split:], indices[:split]
    
    train_dataset = Subset(full_dataset, train_idx)
    train_dataset.dataset.transform = train_transform
    
    val_dataset = Subset(full_dataset, val_idx)
    val_dataset.dataset.transform = val_transform
    
    # labels of the subset being used (epsodic sampler)
    train_labels = [full_dataset.targets[i] for i in train_idx]
    val_labels = [full_dataset.targets[i] for i in val_idx]
    
    # check we have enough classes
    unique_train_classes = len(set(train_labels))
    assert unique_train_classes >= args.n_way, f"Train split only has {unique_train_classes} classes, but n-way is {args.n_way}"
    
    train_sampler = EpisodicBatchSampler(train_labels, args.n_way, args.k_shot, args.q_query, args.episodes)
    val_sampler = EpisodicBatchSampler(val_labels, args.n_way, args.k_shot, args.q_query, args.val_episodes)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0, pin_memory=True)
    
    # Model
    model = PrototypicalNet(backbone_name=args.backbone, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, args.n_way, args.k_shot, args.q_query, device)
        val_loss, val_acc = eval_epoch(model, val_loader, args.n_way, args.k_shot, args.q_query, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.out_dir, f"proto_{args.backbone}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}!")

    print(f"\nTraining complete. Validation Accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
