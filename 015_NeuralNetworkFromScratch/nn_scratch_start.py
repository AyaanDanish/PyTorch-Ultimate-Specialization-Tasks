
#%% packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#%% data prep
# source: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
df = pd.read_csv('heart.csv')
df.head()

#%% separate independent / dependent features
X = np.array(df.loc[ :, df.columns != 'output'])
y = np.array(df['output'])

print(f"X: {X.shape}, y: {y.shape}")

#%% Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#%% scale the data
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

#%% network class
class NeuralNetworkFromScratch:
    def __init__(self, LR, X_train_scale, y_train, X_test_scale, y_test):
        self.W = np.random.randn(X_train_scale.shape[1])
        self.b = np.random.randn()
        self.LR = LR
        self.X_train = X_train_scale
        self.y_train = y_train
        self.X_test = X_test_scale
        self.y_test = y_test
        self.loss_train = []
        self.loss_test = []
    
    def activation(self, z):
        return 1 / (1 + np.exp(-z))
    
    def d_activation(self, z):
        return self.activation(z) * (1 - self.activation(z))
    
    def forward(self, X):
        hidden_1 = np.dot(X, self.W) + self.b
        activation_1 = self.activation(hidden_1)
        return activation_1
    
    def backward(self, X, y_true):
        hidden_1 = np.dot(X, self.W) + self.b
        y_pred = self.forward(X)
        dL_dpred = 2*y_pred - 2*y_true
        d_pred_d_hidden = self.d_activation(hidden_1)
        dhidden1_db = 1
        dhidden1_dW = X

        dl_db = dL_dpred * d_pred_d_hidden * dhidden1_db 
        dl_dw = dL_dpred * d_pred_d_hidden * dhidden1_dW
        return dl_db, dl_dw

    def optimizer(self, dl_db, dl_dw):
        self.b -= self.LR * dl_db
        self.W -= self.LR * dl_dw

    def train(self, ITERATIONS):
        for i in range(ITERATIONS):
            # Pick a random sample
            random_pos = np.random.randint(len(self.X_train))
            X_sample = self.X_train[random_pos]
            y_sample = self.y_train[random_pos]

            # Forward pass
            y_pred = self.forward(X_sample)

            # Compute loss
            loss = np.mean((y_pred - y_sample) ** 2)
            self.loss_train.append(loss)

            # Backward pass
            dL_db, dL_dW = self.backward(X_sample, y_sample)
            self.optimizer(dL_db, dL_dW)

            # Tes t loss calculation
            test_loss = 0
            for j in range(len(self.X_test)):
                y_test_true = self.y_test[j]
                y_test_pred = self.forward(self.X_test[j])
                test_loss += np.mean((y_test_pred - y_test_true) ** 2)
            self.loss_test.append(test_loss / len(self.X_test))

        return 'Training Done'

        
#%% Hyper parameters
LR = 0.5
ITERATIONS = 1000
#%% model instance and training
model= NeuralNetworkFromScratch(LR, X_train_scale, y_train, X_test_scale, y_test)

model.train(ITERATIONS)
# %% check losses
sns.lineplot(x = list(range(len(model.loss_test))), y = model.loss_test)
# %% iterate over test data
total = X_test_scale.shape[0]
correct = 0
y_preds = []

for i in range(total):
    y_true = y_test[i]
    y_pred = np.round(model.forward(X_test_scale[i]))
    y_preds.append(y_pred)
    correct += y_pred == y_true 

# %% Calculate Accuracy
correct / total
# %% Baseline Classifier
from collections import Counter
Counter(y_test)


# %% Confusion Matrix
confusion_matrix(y_test, y_preds)

# %%
sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, cmap='Blues')
# %%
