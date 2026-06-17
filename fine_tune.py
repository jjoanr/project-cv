import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from train_few_shot import (
    PrototypicalNet, 
    EpisodicBatchSampler, 
    train_epoch, 
    eval_epoch, 
    set_seed
)

def extract_prototypes(model, dataset, indices, device, batch_size=32):
    """Extracts the final central prototypes using the exact training images without augmentations"""
    model.eval()
    class_features = {c: [] for c in range(len(dataset.classes))}
    
    temp_subset = Subset(dataset, indices)
    temp_subset.dataset.transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    loader = DataLoader(temp_subset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            features = model.backbone(imgs)
            for i in range(len(labels)):
                class_features[labels[i].item()].append(features[i].cpu())
                
    prototypes = []
    for c in range(len(dataset.classes)):
        stacked = torch.stack(class_features[c])
        prototypes.append(stacked.mean(dim=0))
        
    return torch.stack(prototypes).to(device)

def main():
    parser = argparse.ArgumentParser(description="Master Transfer Learning Pipeline")
    parser.add_argument("--data-dir", required=True, help="Path to external dataset")
    parser.add_argument("--weights", required=True, help="Path to base .pth model")
    parser.add_argument("--images-per-class", type=int, required=True, help="10, 30, or 150")
    parser.add_argument("--unfreeze", choices=["layer4", "layer34", "all"], required=True, help="Which layers to unfreeze")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--out-dir", default="results", help="Directory to save weights and plots")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- EXPERIMENT START: {args.images_per_class} Images | Unfreeze: {args.unfreeze} ---")
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(192, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(args.data_dir)
    classes = full_dataset.classes
    
    # Split data
    class_indices = {i: [] for i in range(len(classes))}
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)
        
    train_idx, val_idx = [], []
    train_counts, val_counts = [], []
    
    for cls, idxs in class_indices.items():
        random.shuffle(idxs)
        total = len(idxs)
        
        train_count = min(args.images_per_class, total - 2)
        
        if train_count < 2:
            train_count = max(2, int(total * 0.8))
            
        train_idx.extend(idxs[:train_count])
        val_idx.extend(idxs[train_count:])
        
        train_counts.append(train_count)
        val_counts.append(total - train_count)
        
    print(f"Training on {len(train_idx)} images. Holding out {len(val_idx)} images for evaluation.")
    
    # episode scaling (bsaed on available images)
    min_train = min(train_counts)
    min_val = min(val_counts)
    safe_pool = min(min_train, min_val)
    
    # Automatically adjust shots
    k_shot = min(5, max(1, safe_pool // 2))
    q_query = min(5, max(1, safe_pool - k_shot))
    
    print(f"Auto-adjusted episodic sampler: {k_shot}-shot, {q_query}-query (Based on bottleneck class)")
    
    train_episodes = 50 if args.images_per_class > 15 else 30
    
    train_dataset = Subset(full_dataset, train_idx)
    train_dataset.dataset.transform = train_transform
    val_dataset = Subset(full_dataset, val_idx)
    val_dataset.dataset.transform = val_transform
    
    train_labels = [full_dataset.targets[i] for i in train_idx]
    val_labels = [full_dataset.targets[i] for i in val_idx]
    
    train_sampler = EpisodicBatchSampler(train_labels, len(classes), k_shot, q_query, train_episodes)
    val_sampler = EpisodicBatchSampler(val_labels, len(classes), k_shot, q_query, 20)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0)
    
    # Load and config model
    model = PrototypicalNet(backbone_name="resnet18", pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    
    for name, param in model.backbone.named_parameters():
        if args.unfreeze == "all":
            param.requires_grad = True
        elif args.unfreeze == "layer34":
            param.requires_grad = True if ("layer3" in name or "layer4" in name) else False
        elif args.unfreeze == "layer4":
            param.requires_grad = True if "layer4" in name else False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Training loop
    best_val_acc = 0.0
    best_model_path = os.path.join(args.out_dir, f"model_{args.images_per_class}img_{args.unfreeze}.pth")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, len(classes), k_shot, q_query, device)
        val_loss, val_acc = eval_epoch(model, val_loader, len(classes), k_shot, q_query, device)
        print(f"Epoch {epoch}/{args.epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            
    # Evalation and conf. matrix
    print("\n--- Generating Final Confusion Matrix on Holdout Set ---")
    model.load_state_dict(torch.load(best_model_path))
    
    # Use the training images to build the final prototypes
    prototypes = extract_prototypes(model, full_dataset, train_idx, device)
    
    # Evaluate on the validation set
    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    all_preds, all_trues = [], []
    
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            features = model.backbone(imgs)
            
            # Calculate euclidean distance to each prototype
            dists = torch.cdist(features, prototypes)
            preds = torch.argmin(dists, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(labels.cpu().numpy())
            
    print("\n" + classification_report(all_trues, all_preds, target_names=classes))
    
    # plot and save conf. matrix
    cm = confusion_matrix(all_trues, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Holdout Confusion Matrix: {args.images_per_class} Images/Class ({args.unfreeze} unfrozen)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plot_path = os.path.join(args.out_dir, f"cm_{args.images_per_class}img_{args.unfreeze}.png")
    plt.savefig(plot_path)
    print(f"Matrix saved to: {plot_path}")
    print("--- EXPERIMENT COMPLETE ---\n")

if __name__ == '__main__':
    main()
