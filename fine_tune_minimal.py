import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from train_few_shot import (
    PrototypicalNet, 
    EpisodicBatchSampler, 
    train_epoch, 
    eval_epoch, 
    set_seed
)

def main():
    parser = argparse.ArgumentParser(description="Minimal Data Fine-Tuning (Feature Extraction)")
    parser.add_argument("--data-dir", required=True, help="Path to external dataset")
    parser.add_argument("--weights", required=True, help="Path to your best .pth model file")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--n-way", type=int, default=13)
    parser.add_argument("--k-shot", type=int, default=3, help="Lower shot for tiny dataset")
    parser.add_argument("--q-query", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--val-episodes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--images-per-class", type=int, default=30, help="STRICT CAP: 10 images per class")
    parser.add_argument("--out-dir", default="model")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Standard transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(192, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(args.data_dir)
    
    # --- Strict Manual Split: Exactly 10 Training Images Per Class ---
    class_indices = {i: [] for i in range(len(full_dataset.classes))}
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)
        
    train_idx = []
    val_idx = []
    
    for cls, idxs in class_indices.items():
        random.shuffle(idxs)
        # Take exactly 10 for training
        train_idx.extend(idxs[:args.images_per_class])
        # The rest go to validation so we have a pure, leak-free test set
        val_idx.extend(idxs[args.images_per_class:])
        
    print(f"Strictly capped training set to {len(train_idx)} images total ({args.images_per_class} per class).")
    print(f"Held out {len(val_idx)} unseen images for pure validation.")
    
    train_dataset = Subset(full_dataset, train_idx)
    train_dataset.dataset.transform = train_transform
    val_dataset = Subset(full_dataset, val_idx)
    val_dataset.dataset.transform = val_transform
    
    train_labels = [full_dataset.targets[i] for i in train_idx]
    val_labels = [full_dataset.targets[i] for i in val_idx]
    
    train_sampler = EpisodicBatchSampler(train_labels, args.n_way, args.k_shot, args.q_query, args.episodes)
    val_sampler = EpisodicBatchSampler(val_labels, args.n_way, args.k_shot, args.q_query, args.val_episodes)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0)
    
    # Load model architecture
    model = PrototypicalNet(backbone_name=args.backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    
    # --- FEATURE EXTRACTION: Freeze Early Layers ---
    print("Freezing early ResNet layers (Feature Extraction Mode)...")
    for name, param in model.backbone.named_parameters():
        # Freeze everything EXCEPT layer 4 (the deep semantic filters)
        if "layer4" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            
    # Only optimize parameters that are unfrozen
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    best_val_acc = 0.0
    print("\nStarting Minimal-Data Fine-Tuning...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, args.n_way, args.k_shot, args.q_query, device)
        val_loss, val_acc = eval_epoch(model, val_loader, args.n_way, args.k_shot, args.q_query, device)
        
        print(f"Epoch {epoch}/{args.epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.out_dir, "proto_resnet18_minimal_data.pth")
            torch.save(model.state_dict(), save_path)
            
    print(f"\nTraining Complete! Best Leak-Free Val Accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
