# Libraries
import torch
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Subset
from metrics_functions import accuracy_fn, recall_fn, precision_fn, f1_fn


### 1. PREPROCESSING DATA
## Transformations, downloading train & test sets
transform = tv.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # (mean, std) for RGB -> 3 colour channels
])

trainset = tv.datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transform,
    download=True 
)

testset = tv.datasets.CIFAR10(
    root="./data",
    train=False,
    transform=transform,
    download=True
)


image_classes = [0,8]  # For airplane & ship


## Selecting specific classes
tr_indices = [i for i, label in enumerate (trainset.targets) if label in image_classes]
train = Subset(trainset, tr_indices)

test_indices = [i for i, label in enumerate (testset.targets) if label in image_classes]
test = Subset(testset, test_indices)


## Train-test split
X_train, y_train, X_test, y_test = [], [], [], []
for image, label in train:
    X_train.append(image)
    y_train.append(label)

for image, label in test:
    X_test.append(image)
    y_test.append(label)

X_train, X_test = torch.stack(X_train), torch.stack(X_test)


# Remap labels
y_train, y_test = torch.tensor([0 if label == 0 else 1 for label in y_train]), torch.tensor([0 if label == 0 else 1 for label in y_test])


## Creating Dataloaders, splitting into batches
train_set, test_set = TensorDataset(X_train, y_train), TensorDataset(X_test, y_test)
train_loader, test_loader = DataLoader(train_set, batch_size=64, shuffle=True), DataLoader(test_set, batch_size=64)

print("Unique y_train:", torch.unique(y_train, return_counts=True))
print("Unique y_test:", torch.unique(y_test, return_counts=True))


# Libraries
from torch import nn



### 2. BUILDING A MODEL
## Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


## Model
modelResNetV0 = tv.models.resnet18(weights=None).to(device)
modelResNetV0.fc = nn.Linear(modelResNetV0.fc.in_features, 2).to(device)
modelResNetV1 = tv.models.resnet50(weights=None).to(device)
modelResNetV1.fc = nn.Linear(modelResNetV1.fc.in_features, 2).to(device)


# Loss function & optimizer
loss_fn = nn.CrossEntropyLoss()
optimizerV0 = torch.optim.Adam(modelResNetV0.parameters(), lr=0.001)
optimizerV1 = torch.optim.Adam(modelResNetV1.parameters(), lr=0.001)



### 3. TRAINING LOOP
torch.manual_seed(42)
epochs = 15


# Loop
for epoch in range (epochs):
    ## Training
    modelResNetV0.train()
    modelResNetV1.train()
    train_loss_V0, train_acc_V0, train_loss_V1, train_acc_V1 = 0, 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        y_logits_V0 = modelResNetV0(X_batch)
        y_preds_V0 = torch.softmax(y_logits_V0, dim=1).argmax(dim=1)

        y_logits_V1 = modelResNetV1(X_batch)
        y_preds_V1 = torch.softmax(y_logits_V1, dim=1).argmax(dim=1)

        # Loss and acc
        loss_V0 = loss_fn(y_logits_V0, y_batch)
        loss_V1 = loss_fn(y_logits_V1, y_batch)

        optimizerV0.zero_grad()
        loss_V0.backward()
        optimizerV0.step()
        train_loss_V0 += loss_V0.item()
        train_acc_V0 += accuracy_fn(y_preds_V0, y_batch)

        optimizerV1.zero_grad()
        loss_V1.backward()
        optimizerV1.step()
        train_loss_V1 += loss_V1.item()
        train_acc_V1 += accuracy_fn(y_preds_V1, y_batch)


    # Testing
    modelResNetV0.eval()
    modelResNetV1.eval()
    all_test_logits_V0, all_test_logits_V1, all_test_labels = [], [], []
    test_loss_V0, test_acc_V0, test_loss_V1, test_acc_V1 = 0, 0, 0, 0
    with torch.inference_mode():
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            test_logits_V0 = modelResNetV0(X_batch)
            test_preds_V0 = torch.softmax(test_logits_V0, dim=1).argmax(dim=1)

            test_logits_V1 = modelResNetV1(X_batch)
            test_preds_V1 = torch.softmax(test_logits_V1, dim=1).argmax(dim=1)

            # Loss and acc
            t_loss_V0 = loss_fn(test_logits_V0, y_batch)
            t_loss_V1 = loss_fn(test_logits_V1, y_batch)

            test_loss_V0 += t_loss_V0.item()
            test_acc_V0 += accuracy_fn(test_preds_V0, y_batch)

            test_loss_V1 += t_loss_V1.item()
            test_acc_V1 += accuracy_fn(test_preds_V1, y_batch)

            ## Append for final metrics
            all_test_logits_V0.append(test_logits_V0.cpu())
            all_test_logits_V1.append(test_logits_V1.cpu())
            all_test_labels.append(y_batch.cpu())


    
    print(f"Epoch: {epoch} | \n"
          f"Train Loss V0: {train_loss_V0/len(train_loader):.4f} | "
          f"Train Acc V0: {train_acc_V0/len(train_loader):.2f}% | "
          f"Test Loss V0: {test_loss_V0/len(test_loader):.4f} | "
          f"Test Acc V0: {test_acc_V0/len(test_loader):.2f}%")

    print(f"Train Loss V1: {train_loss_V1/len(train_loader):.4f} | "
          f"Train Acc V1: {train_acc_V1/len(train_loader):.2f}% | "
          f"Test Loss V1: {test_loss_V1/len(test_loader):.4f} | "
          f"Test Acc V1: {test_acc_V1/len(test_loader):.2f}%")


# Find metrics
all_test_logits_V0 = torch.cat(all_test_logits_V0)
all_test_logits_V1 = torch.cat(all_test_logits_V1)
all_test_labels = torch.cat(all_test_labels)


print(f"Test Precision V0: {precision_fn(all_test_logits_V0, all_test_labels):.2f}% | "
      f"Test Precision V1: {precision_fn(all_test_logits_V1, all_test_labels):.2f}% | "
      f"Test Recall V0: {recall_fn(all_test_logits_V0, all_test_labels):.2f}% | "
      f"Test Recall V1: {recall_fn(all_test_logits_V1, all_test_labels):.2f}% | "
      f"Test F1 V0: {f1_fn(all_test_logits_V0, all_test_labels):.2f}% | "
      f"Test F1 V1: {f1_fn(all_test_logits_V1, all_test_labels):.2f}% | " )


### Saving for future use
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True, parents=True)


models = {
    "cifarResNetV0.pt": modelResNetV0,
    "cifarResNetV1.pt": modelResNetV1
}

for name, model in models.items():
    MODEL_SAVE_PATH = MODEL_PATH / name
    print(f"Saving model {name} to {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)