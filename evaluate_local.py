import argparse
from pathlib import Path
import random
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import your network architecture
from train_few_shot import PrototypicalNet, get_euclidean_distances

def get_transform():
    return transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

@torch.no_grad()
def evaluate_holdout(weights_path, backbone, test_dir, shots, device, seed=42):
    random.seed(seed)
    
    # 1. Load Model
    model = PrototypicalNet(backbone_name=backbone, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    transform = get_transform()
    test_path = Path(test_dir)
    classes = sorted([d.name for d in test_path.iterdir() if d.is_dir()])
    
    prototypes = []
    queries = []
    y_true = []
    
    print(f"Dynamically splitting '{test_dir}' into {shots}-shot support and query sets...")
    
    valid_classes = [] # <-- NEW: Keep track of successfully loaded classes
    
    # 2. Build Prototypes and Query List dynamically
    for cls in classes:
        all_images = [p for p in (test_path / cls).iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        if len(all_images) <= shots:
            print(f"Skipping {cls}: Not enough images (needs > {shots}).")
            continue
            
        random.shuffle(all_images)
        support_imgs = all_images[:shots]
        query_imgs = all_images[shots:]
        
        # Calculate prototype for this class
        embeddings = [model(transform(Image.open(img).convert("RGB")).unsqueeze(0).to(device)) for img in support_imgs]
        prototypes.append(torch.cat(embeddings).mean(dim=0))
        valid_classes.append(cls) # <-- NEW: Add to our valid list
        
        # Store queries and their ground truth labels
        for q_img in query_imgs:
            queries.append(q_img)
            y_true.append(cls)
            
    if not prototypes:
        print("No prototypes could be built. Check your test directory.")
        return
        
    prototypes = torch.stack(prototypes)
    y_pred = []
    
    # 3. Evaluate Queries
    print(f"Evaluating {len(queries)} unseen physical images...")
    for crop_path in tqdm(queries):
        img = Image.open(crop_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        
        dists = get_euclidean_distances(model(tensor), prototypes)
        # <-- NEW: Use valid_classes instead of the original classes list
        prediction = valid_classes[torch.argmin(dists, dim=1).item()] 
        y_pred.append(prediction)
            
    # 4. Print Report and Save Confusion Matrix
    print("\n--- Final Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=valid_classes))
    
    cm = confusion_matrix(y_true, y_pred, labels=valid_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=valid_classes, yticklabels=valid_classes)
    plt.title('Final Holdout Evaluation - Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = 'final_confusion_matrix.png'
    plt.savefig(output_path, dpi=300)
    print(f"\nConfusion matrix saved to {output_path}!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to your downloaded .pth model file")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--test-dir", required=True, help="Path to your DATASET/local_test directory")
    parser.add_argument("--shots", type=int, default=5, help="Number of images per class to use as support")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_holdout(args.weights, args.backbone, args.test_dir, args.shots, device)
