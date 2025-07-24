import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path

# --- Configuration ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_SET_RATIO = 0.10
VAL_SET_RATIO = 0.10
NUM_EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.001
ITERATIONS = 20
OUTPUT_DIR_RANDOM = Path("output_entropy_base/random")
OUTPUT_DIR_WISE = Path("output_entropy_base/entropy")

# --- Determinism ---
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Output Directories ---
OUTPUT_DIR_RANDOM.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_WISE.mkdir(parents=True, exist_ok=True)

# --- Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        return F.relu(self.fc1(x))

# --- Helper Functions ---
def train(model, train_loader, val_loader, optimizer, criterion, device):
    best_val_f1 = -1
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        val_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()

    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return best_val_f1

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    return f1_score(all_labels, all_preds, average="macro", zero_division=0)

def select_by_entropy(pool_loader, model, k, device):
    """
    Selects k samples from the pool based on the highest predictive entropy.

    Entropy-based sampling is a powerful active learning strategy that measures the model's uncertainty.
    - A low entropy value means the model is very confident about one class (e.g., [0.9, 0.05, 0.05]).
    - A high entropy value means the model's predictions are spread out across multiple classes,
      indicating high uncertainty (e.g., [0.33, 0.33, 0.34]).

    Compared to the simpler least-confidence strategy (which only considers the probability of the
    most likely class), entropy considers the entire probability distribution. This allows it to
    better capture samples where the model is confused between several classes, making them highly
    informative for the next training iteration.
    """
    model.eval()
    entropies = []
    
    with torch.no_grad():
        # A full pass over the pool is needed to calculate entropy for each sample.
        for data, _ in pool_loader:
            data = data.to(device)
            output = model(data)
            # Use log_softmax for better numerical stability when calculating entropy.
            log_probs = F.log_softmax(output, dim=1)
            probs = torch.exp(log_probs)
            # The entropy is the negative sum of p(x) * log(p(x))
            entropy = -torch.sum(probs * log_probs, dim=1)
            entropies.extend(entropy.cpu().numpy())

    entropies = np.array(entropies)
    # We want the indices with the highest entropy, so we sort in descending order.
    # np.argsort on the negative entropies gives us the indices from highest to lowest.
    selected_indices = np.argsort(-entropies)[:k]
    
    # Map the selected indices back to their original indices in the full dataset.
    original_indices = [pool_loader.dataset.indices[i] for i in selected_indices]
    
    return original_indices

def plot_tsne(features, labels, title, filename):
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(features)-1))
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(scatter, ticks=range(10))
    plt.title(title)
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.savefig(filename)
    plt.close()

def main():
    # --- Data Loading and Splitting ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Create indices for splits
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    
    test_split_idx = int(num_train * TEST_SET_RATIO)
    val_split_idx = test_split_idx + int(num_train * VAL_SET_RATIO)
    
    test_indices = indices[:test_split_idx]
    val_indices = indices[test_split_idx:val_split_idx]
    unlabeled_pool_indices = indices[val_split_idx:]

    test_set = Subset(full_train_dataset, test_indices)
    val_set = Subset(full_train_dataset, val_indices)
    unlabeled_pool = Subset(full_train_dataset, unlabeled_pool_indices)

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Test set size: {len(test_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Unlabeled pool size: {len(unlabeled_pool)}")

    # --- Phase 1: Baseline Evaluation ---
    print("\n--- Starting Baseline Evaluation ---")
    baseline_model = SimpleCNN().to(DEVICE)
    baseline_test_f1 = evaluate(baseline_model, test_loader, DEVICE)
    baseline_val_f1 = evaluate(baseline_model, val_loader, DEVICE)
    
    results = {
        "Percentage": [0], 
        "Random Test F1": [baseline_test_f1], 
        "Wise Test F1": [baseline_test_f1],
        "Random Val F1": [baseline_val_f1],
        "Wise Val F1": [baseline_val_f1]
    }

    # Generate t-SNE for baseline
    baseline_model.eval()
    test_features = []
    test_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(DEVICE)
            features = baseline_model.get_features(data)
            test_features.append(features.cpu().numpy())
            test_labels.append(target.cpu().numpy())

    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    tsne_title = f"Baseline (Untrained) – Test F1={baseline_test_f1:.4f}"
    plot_tsne(test_features, test_labels, tsne_title, OUTPUT_DIR_RANDOM / "tsne_baseline.png")
    plot_tsne(test_features, test_labels, tsne_title, OUTPUT_DIR_WISE / "tsne_baseline.png")
    print(f"Baseline Test F1: {baseline_test_f1:.4f}. t-SNE plots saved.")

    # --- Phase 2: Iterative Training Loop ---
    print("\n--- Starting Iterative Training ---")
    
    # Initialize models and optimizers once before the loop
    model_random = SimpleCNN().to(DEVICE)
    optimizer_random = optim.Adam(model_random.parameters(), lr=LEARNING_RATE)
    
    model_wise = SimpleCNN().to(DEVICE)
    optimizer_wise = optim.Adam(model_wise.parameters(), lr=LEARNING_RATE)

    wise_selection_model = SimpleCNN().to(DEVICE)
    wise_selection_model.load_state_dict(baseline_model.state_dict())

    acquired_indices_random = []
    acquired_indices_wise = []
    
    remaining_pool_indices_random = list(unlabeled_pool_indices)
    remaining_pool_indices_wise = list(unlabeled_pool_indices)

    for p in tqdm(range(1, ITERATIONS + 1), desc="Active Learning Iterations"):
        target_size = int(p / 100 * len(unlabeled_pool))
        
        # --- Random Sampling ---
        num_new_samples_random = target_size - len(acquired_indices_random)
        if num_new_samples_random > 0:
            new_indices = np.random.choice(remaining_pool_indices_random, size=num_new_samples_random, replace=False)
            acquired_indices_random.extend(new_indices)
            remaining_pool_indices_random = [idx for idx in remaining_pool_indices_random if idx not in new_indices]

        train_set_random = Subset(full_train_dataset, acquired_indices_random)
        train_loader_random = DataLoader(train_set_random, batch_size=BATCH_SIZE, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()

        print(f"Training Random model with {len(train_set_random)} samples...")
        val_f1_random = train(model_random, train_loader_random, val_loader, optimizer_random, criterion, DEVICE)
        test_f1_random = evaluate(model_random, test_loader, DEVICE)
        torch.save(model_random.state_dict(), OUTPUT_DIR_RANDOM / f"model_iter_{p:02d}_val_f1_{val_f1_random:.4f}.pth")

        # t-SNE for random
        model_random.eval()
        test_features_rand = []
        with torch.no_grad():
            for data, _ in test_loader:
                features = model_random.get_features(data.to(DEVICE))
                test_features_rand.append(features.cpu().numpy())
        test_features_rand = np.concatenate(test_features_rand, axis=0)
        tsne_title_rand = f"Random – {p}% Trained – Test F1={test_f1_random:.4f}"
        plot_tsne(test_features_rand, test_labels, tsne_title_rand, OUTPUT_DIR_RANDOM / f"tsne_p{p:02d}.png")


        # --- Wise (Entropy-Based) Sampling ---
        num_new_samples_wise = target_size - len(acquired_indices_wise)
        if num_new_samples_wise > 0:
            if p == 1:
                print("Wise selection model is untrained. Using random sampling for the first iteration.")
                new_indices_wise = np.random.choice(remaining_pool_indices_wise, size=num_new_samples_wise, replace=False)
            else:
                # Create a loader for the remaining pool to select from
                remaining_pool_wise_subset = Subset(full_train_dataset, remaining_pool_indices_wise)
                pool_loader_wise = DataLoader(remaining_pool_wise_subset, batch_size=BATCH_SIZE, shuffle=False)
                
                new_indices_wise = select_by_entropy(pool_loader_wise, wise_selection_model, num_new_samples_wise, DEVICE)

            acquired_indices_wise.extend(new_indices_wise)
            remaining_pool_indices_wise = [idx for idx in remaining_pool_indices_wise if idx not in new_indices_wise]

        train_set_wise = Subset(full_train_dataset, acquired_indices_wise)
        train_loader_wise = DataLoader(train_set_wise, batch_size=BATCH_SIZE, shuffle=True)
        
        print(f"Training Wise model with {len(train_set_wise)} samples...")
        val_f1_wise = train(model_wise, train_loader_wise, val_loader, optimizer_wise, criterion, DEVICE)
        test_f1_wise = evaluate(model_wise, test_loader, DEVICE)
        torch.save(model_wise.state_dict(), OUTPUT_DIR_WISE / f"model_iter_{p:02d}_val_f1_{val_f1_wise:.4f}.pth")

        # Update the model for the next wise selection step
        wise_selection_model.load_state_dict(model_wise.state_dict())

        # t-SNE for wise
        model_wise.eval()
        test_features_wise = []
        with torch.no_grad():
            for data, _ in test_loader:
                features = model_wise.get_features(data.to(DEVICE))
                test_features_wise.append(features.cpu().numpy())
        test_features_wise = np.concatenate(test_features_wise, axis=0)
        tsne_title_wise = f"Wise (Entropy) – {p}% Trained – Test F1={test_f1_wise:.4f}"
        plot_tsne(test_features_wise, test_labels, tsne_title_wise, OUTPUT_DIR_WISE / f"tsne_p{p:02d}.png")
        
        results["Percentage"].append(p)
        results["Random Test F1"].append(test_f1_random)
        results["Wise Test F1"].append(test_f1_wise)
        results["Random Val F1"].append(val_f1_random)
        results["Wise Val F1"].append(val_f1_wise)
        
        # --- Print current results ---
        df = pd.DataFrame(results)
        print("\n--- Current F1 Score Summary ---")
        print(df)


    # --- Final Summary ---
    print("\n--- Final F1 Score Summary ---")
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR_WISE / "results_entropy.csv", index=False)
    print(df)

if __name__ == '__main__':
    main() 