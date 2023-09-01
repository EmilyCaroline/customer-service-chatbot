# Import necessary libraries
import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load intents from JSON file
with open('sample.json', 'r') as f:
    intents = json.load(f)

# Extract data from intents
all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X = []
y = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X.append(bag)
    label = tags.index(tag)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ... (Rest of your code, including dataset setup, model creation, and training)
# Hyper-parameters 
num_epochs = 200
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

print('Befor trainning')
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    print(epoch)
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]data
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted_train = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted_train.eq(labels).sum().item()

    train_losses.append(train_loss / len(train_loader))
    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}')

class TestChatDataset(Dataset):

    def __init__(self):
        self.n_samples_test = len(X_test)
        self.x_test_data = X_test
        self.y_test_data = y_test

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, idx):
        return self.x_test_data[idx], self.y_test_data[idx]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples_test

dataset = TestChatDataset()
test_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0)

# Evaluate the model on test data
model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0
with torch.no_grad():
    for words, labels in test_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        _, predicted_test = outputs.max(1)
        total_test += labels.size(0)
        correct_test += predicted_test.eq(labels).sum().item()

        test_losses.append(test_loss / len(test_loader))
        test_accuracy = correct_test / total_test
        test_accuracies.append(test_accuracy)

# Print final accuracies
print(f'Final Training Accuracy: {train_accuracies[-1]:.4f}')
print(f'Final Test Accuracy: {test_accuracies[-1]:.4f}')

print("Test Loss")
print(test_losses)
print("Train Loss")

print(train_losses)

plt.show()
# Plot accuracy vs loss for training and testing data
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Accuracy vs Loss - Training and Testing Data')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(test_accuracies, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Loss - Training and Testing Data')
plt.legend()

plt.show()

# Evaluate the model on training data
model.eval()
with torch.no_grad():
    y_true_train = []
    y_pred_train = []
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        outputs = model(words)
        predicted_labels = torch.argmax(outputs, dim=1)
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(predicted_labels.cpu().numpy())

# Calculate and print accuracy for training data
accuracy_train = accuracy_score(y_true_train, y_pred_train)
print(f'Training Accuracy: {accuracy_train:.4f}')

# Create confusion matrix for training data
conf_matrix_train = confusion_matrix(y_true_train, y_pred_train, labels=range(output_size))

# Plot confusion matrix for training data
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', xticklabels=tags, yticklabels=tags)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Training Data')
plt.show()

class TestChatDataset(Dataset):

    def __init__(self):
        self.n_samples_test = len(X_test)
        self.x_test_data = X_test
        self.y_test_data = y_test

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, idx):
        return self.x_test_data[idx], self.y_test_data[idx]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples_test

dataset = TestChatDataset()
test_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0)

# Evaluate the model on test data
model.eval()
with torch.no_grad():
    y_true_test = []
    y_pred_test = []
    for words, labels in test_loader:
        words = words.to(device)
        labels = labels.to(device)
        outputs = model(words)
        predicted_labels = torch.argmax(outputs, dim=1)
        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(predicted_labels.cpu().numpy())

# Calculate and print accuracy for test data
accuracy_test = accuracy_score(y_true_test, y_pred_test)
print(f'Test Accuracy: {accuracy_test:.4f}')

# Create confusion matrix for test data
conf_matrix_test = confusion_matrix(y_true_test, y_pred_test, labels=range(output_size))

# Plot confusion matrix for test data
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=tags, yticklabels=tags)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Test Data')
plt.show()
