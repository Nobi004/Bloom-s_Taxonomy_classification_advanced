import pandas as pd    
import torch 
import torch.nn as nn     
import torch.optim as optim 
from torch.utils.data import Dataset ,DataLoader 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter 
import numpy as np 
import os      

# Custom Dataset 
class TextDataset(Dataset):
    def __init__(self,texts,labels,vocab,max_len=100):
        self.texts = texts       
        self.labels = labels 
        self.vocab = vocab
        self.max_len = max_len 

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        words = text.split()
        indices = [self.vocab.get(word,self.vocab['<UNK>']) for word in words]
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        else: 
            indices = indices[:self.max_len]
        return torch.tensor(indices), torch.tensor(self.labels[idx])
    

# TextCNN 
class TextCNN(nn.Module):
    def __init__(self,vocab_size,embed_dim,num_classes,kernel_sizes=[3,4,5],num_filters=100):
        super(TextCNN,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1,num_filters, (k,embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self,x):
        x = self.embedding(x).unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(xi,xi.size(2)).squeeze(2) for xi in x]
        x = torch.cat(x,1)
        x = self.dropout(x)
        return self.fc(x)
    
# Load data
train = pd.read_csv('artifacts/data/train.csv')
val = pd.read_csv('artifacts/data/val.csv')
test = pd.read_csv('artifacts/data/test.csv')

# Build vocab
words = ' '.join(train['text']).split()
vocab = {word: i+2 for i, word in enumerate(set(words))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

# Dataset and DataLoader
train_dataset = TextDataset(train['text'].values, train['label'].values, vocab)
val_dataset = TextDataset(val['text'].values, val['label'].values, vocab)
test_dataset = TextDataset(test['text'].values, test['label'].values, vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextCNN(len(vocab), embed_dim=100, num_classes=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    model.train()
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch+1}, Val Acc: {val_acc:.3f}")

# Test
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts)
        test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds, average='weighted')

# Save
os.makedirs('artifacts/models', exist_ok=True)
torch.save(model.state_dict(), 'artifacts/models/textcnn.pt')
np.save('artifacts/results/textcnn_results.npy', {'acc': test_acc, 'f1': test_f1, 'pred': test_preds})

print(f"TextCNN trained. Test Acc: {test_acc:.3f}")