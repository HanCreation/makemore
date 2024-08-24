import torch
import torch.nn as nn
import torch.nn.functional as F

#Hyperparameters
batch_size = 32
block_size=8
max_iters=3000
eval_interval=300 #Do evaluation every eval_interval iterations
learning_rate= 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Use GPU (NVIDA RTX for example) if available
eval_iters=200

# torch.manual_seed(1337)

#Data ('tinyshakespeare.txt' is a tiny version of 'shakespeare.txt')
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
file_name='tinyshakespeare.txt' # change to tinyshakespeare directory
with open(file_name,'r', encoding='utf-8') as f:
    text=f.read()
    
#Vocab Ambil semua karakter unik dalam dataset
chars=sorted(list(set(text)))
vocab_size=len(chars)
# print("".join(chars))
# print(vocab_size)
#Buat tabel untuk mapping karakter ke integer dan sebaliknya
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
#Encode (string->char to int)
encode=lambda s: [stoi[ch] for ch in s]
#Decode (int->char to string)
decode=lambda l: "".join([itos[i] for i in l])

#Train-Val split
#Misahin data menjadi train set dan validation set, supaya model bisa belajar dan diuji, kemudian kita bisa cek ga ada overfitting, ga nginget dataset doang
data=torch.tensor(encode(text), dtype=torch.long)
n= int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

#Data loader
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data # milih split train atau val
    ix = torch.randint(len(data) - block_size, (batch_size,)) # generate random index positions, generate angka random 0 sampai len(data) - block_size sebanyak batch_sizenya, buat offset di training set
    x = torch.stack([data[i:i+block_size] for i in ix]) # index ke i sampai i+block_size, ini inputnya, i itu angka yg ada di array ix
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # targetnya adalah x yang di offset 1
    #torch.stack itu buat numpuk tensor-tensor 1D terus di tumpuk semua (stack them up at rows)
    x, y = x.to(device), y.to(device) # jika ada GPU kalkulasinya bakal kerja di GPU, jadi dipindah ke GPU
    return x, y

@torch.no_grad() #Kasih tau pytorch buat ga nyimpen gradient dari fungsi ini, karena ini buat evaluasi doang dan supaya lebih efisien
def estimate_loss():
    out = {}
    model.eval() # set model ke evaluation mode, karena jika layer layer tertentu bisa punya kelakuan beda saat inference(eval) dan training, contoh kek batchnorm
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # set model ke training mode
    return out

class LanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # setiap token secara langsung baca logits buat next token dari lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        #idx di pass dan bakal baca di embedding table yang di init dan ambil baris yang sesuai dgn idx
        #idx =43 bakal baca baris ke 43 dari embedding table
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None: #Prediction mode
            loss = None
        else:
            B, T, C = logits.shape #Batch, Time, Channel
            #targets itu B x T, kita flatten jadi 1D
            logits = logits.view(B*T, C) #Reshape to 2D supaya cross_entropy kerjanya bener
            targets = targets.view(B*T)
            #Loss function bakal ngebandingin logits dengan target
            loss = F.cross_entropy(logits, targets) #Negative log likelihood (see makemore bigram series)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx adlaah (B x T) tensor dari konteks saat ini
        for _ in range(max_new_tokens):
            # forward pass (Prediction mode)
            logits, loss = self(idx)#Call object dirinya sendiri (forward)
            # ambil idx paling belakang dari T, karena ini adalah prediksi untuk token selanjutnya
            logits = logits[:, -1, :] # becomes (B, C)
            # pasang softmax untuk mendapatkan distribusi probabilitas
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample prediksi dari distribusi probabilitas
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # pasang idx_next yang di prediksi ke sequence yang udah ada buat ngulang lg
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = LanguageModel(vocab_size)
m = model.to(device) # jika ada GPU kalkulasinya bakal kerja di GPU, jadi dipindah ke GPU

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) #Start index
#  [0] itu First dimension/firstbatch for predicition -> m.generate(context, max_new_tokens=500)[0].tolist()
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))