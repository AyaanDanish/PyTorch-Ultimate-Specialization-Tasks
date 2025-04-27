
#%% packages
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from sklearn.manifold import TSNE
import seaborn as sns

# %%
dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())
data = dataset[0]

# %%
from collections import Counter
Counter(data.y.cpu().numpy())

# %%
class CNN(torch.nn.Module):
    def __init__(self, num_hidden, num_features, num_classes):
        super().__init__()
        self.conv1 = GATConv(num_features, num_hidden)
        self.conv2 = GATConv(num_hidden, num_classes)
    
    def forward(self, x, edges_index):
        x = self.conv1(x, edges_index)
        x = x.relu()
        x = F.dropout(x, p=0.2)
        x = self.conv2(x, edges_index)
        return x
    

# %%
model = CNN(num_hidden=16, num_features=dataset.num_features, num_classes=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()
# %%
loss_list = []
model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(data.x, data.edge_index)
    y_true = data.y
    loss = loss_fn(y_pred[data.train_mask], y_true[data.train_mask])
    loss_list.append(loss.item())
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss}")
#%%
sns.lineplot(x=list(range(len(loss_list))), y=loss_list)

# %%
model.eval()
with torch.no_grad():
    y_pred = model(data.x, data.edge_index)
    y_pred_class = y_pred.argmax(dim=1)
    correct = y_pred_class[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())

print('Test Acc:', acc)

# %%
z = TSNE(n_components=2).fit_transform(y_pred[data.test_mask].detach().cpu().numpy())
sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=data.y[data.test_mask])
