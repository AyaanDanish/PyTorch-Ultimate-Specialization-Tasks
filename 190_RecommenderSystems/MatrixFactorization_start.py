#%%
import pandas as pd
from sklearn import model_selection, preprocessing
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.metrics import mean_squared_error
#%% data import
df = pd.read_csv("ratings.csv")
df.head(2)
#%%
print(f"Unique Users: {df.userId.nunique()}, Unique Movies: {df.movieId.nunique()}")

 #%% Data Class
class MovieDataset(Dataset):
    def __init__(self, users, movies, ratings):
        super().__init__()
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)
    def __getitem__(self, idx):
        users = self.users[idx]
        movies = self.movies[idx]
        ratings = self.ratings[idx]
        return torch.tensor(users, dtype=torch.long), torch.tensor(movies, dtype=torch.long), torch.tensor(ratings, dtype=torch.float32)
        

#%% Model Class
class RecSysModel(nn.Module):
    def __init__(self, n_users, n_movies, n_embeddings = 32):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, n_embeddings)
        self.movie_embed = nn.Embedding(n_movies, n_embeddings)
        self.fc1 = nn.Linear(in_features=n_embeddings*2, out_features=1)
    
    def forward(self, users, movies):
        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        x = torch.cat([user_embeds, movie_embeds], dim=1)
        x = self.fc1(x)
        return x
    
    

#%% encode user and movie id to start from 0 
lbl_user = preprocessing.LabelEncoder()
lbl_movie = preprocessing.LabelEncoder()
df['userId'] = lbl_user.fit_transform(df.userId.values)
df['movieId'] = lbl_movie.fit_transform(df.movieId.values)


#%% create train test split
df_train, df_test = model_selection.train_test_split(df, test_size=0.2, random_state=123)

 
#%% Dataset Instances
train_dataset = MovieDataset(
    users=df_train.userId.values,
    movies=df_train.movieId.values,
    ratings=df_train.rating.values
)

valid_dataset = MovieDataset(
    users=df_test.userId.values,
    movies=df_test.movieId.values,
    ratings=df_test.rating.values
)

#%% Data Loaders
BATCH_SIZE = 4
train_loader = DataLoader(dataset=train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True
) 

test_loader = DataLoader(dataset=valid_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True
) 
#%% Model Instance, Optimizer, and Loss Function
model = RecSysModel(
    n_users=len(lbl_user.classes_),
    n_movies=len(lbl_movie.classes_))

optimizer = torch.optim.Adam(model.parameters())  
criterion = nn.MSELoss()
#%% Model Training
NUM_EPOCHS = 1

model.train() 
for epoch_i in range(NUM_EPOCHS):
    for users, movies, ratings in train_loader:
        optimizer.zero_grad()
        y_pred = model(users, 
                       movies)         
        y_true = ratings.unsqueeze(dim=1).to(torch.float32)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

#%% Model Evaluation 
y_preds = []
y_trues = []

model.eval()
with torch.no_grad():
    for users, movies, ratings in test_loader: 
        y_true = ratings.detach().numpy().tolist()
        y_pred = model(users, movies).squeeze().detach().numpy().tolist()
        y_trues.append(y_true)
        y_preds.append(y_pred)

mse = mean_squared_error(y_trues, y_preds)
print(f"Mean Squared Error: {mse}")
#%% Users and Items
user_movie_test = defaultdict(list)
with torch.no_grad():
    for users, movies, ratings in test_loader: 
        y_pred = model(users,movies)
        for i, _ in enumerate(users):
            user_id = users[i].item()
            movie_id = movies[i].item()
            pred_rating = y_pred[i][0].item()
            rating = ratings[i].item()

        print(f"User: {user_id}, Movie: {movie_id}, Rating: {rating}, Prediction: {pred_rating:.1f}")
        user_movie_test[user_id].append((pred_rating, rating))
#%% Precision and Recall
precision = {}
recalls = {}
k = 10

threshold = 3.5
for uid, user_ratings in user_movie_test.items():
    user_ratings.sort(key=lambda x: x[0], reverse=True)

    #count the number of relevant items
    n_rel = sum((true_rating >= threshold) for _, true_rating in user_ratings)

    #count the number of recommended items in top k
    n_rec_k = sum((pred_rating >= threshold) for pred_rating, _ in user_ratings[:k]) 

    #count recommended AND relevant items
    n_rel_and_rec_k = sum(((true_rating >= threshold) and (pred_rating >= threshold)) for pred_rating, true_rating in user_ratings[:k])

    print(f"User: {uid}, Relevant Items: {n_rel}, Recommended Items: {n_rec_k}, Relevant and Recommended Items: {n_rel_and_rec_k}")

    precision[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
    recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

# %%
print(f"Mean Precision: {sum(precision.values()) / len(precision)}")
print(f"Mean Recall: {sum(recalls.values()) / len(recalls)}")
# %%
