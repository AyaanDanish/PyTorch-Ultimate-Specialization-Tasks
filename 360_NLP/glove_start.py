#%%
import torch
import torchtext.vocab as vocab


# %%
glove = vocab.GloVe(name='6B', dim=100)
# %%
glove.vectors.shape
#%%
def get_embedding_vector(word):
    word_index = glove.stoi[word]
    emb = glove.vectors[word_index]
    return emb

# %%
get_embedding_vector('chess').shape
# %%
def get_closest_word_from_word(word, max_n=5):
    word_emb = get_embedding_vector(word)
    distances = [(w, torch.dist(word_emb, get_embedding_vector(w)).cpu().item()) for w in glove.itos]
    return sorted(distances, key=lambda x: x[1])[:max_n]

def get_closest_word_emb(word_emb, max_n=5):
    distances = [(w, torch.dist(word_emb, get_embedding_vector(w)).cpu().item()) for w in glove.itos]
    return sorted(distances, key=lambda x: x[1])[:max_n]

# %%
get_closest_word_from_word('man')
# %%
def get_analogy(word1, word2, word3, max_n=5):
    word4_emb =  get_embedding_vector(word1) - get_embedding_vector(word2) + get_embedding_vector(word3)
    return get_closest_word_emb(word4_emb)
# %%
get_analogy('sister', 'brother', 'nephew')
# %%
