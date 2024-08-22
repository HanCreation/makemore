import torch
import torch.nn as nn
import torch.nn.functional as F

#Hyperparameters
batch_size = 32
block_size=8
max_iters=5000
eval_interval=300 #Do evaluation every eval_interval iterations
learning_rate= 1e-3 #Self-attention ga terlalu bagus dengan learning rate yang tinggi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Use GPU (NVIDA RTX for example) if available
eval_iters=200
n_embd = 32 #Embedding dimensionsize

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

#One head self-attention
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        #Key
        self.key=nn.Linear(n_embd, head_size,bias=False)
        #Query
        self.query=nn.Linear(n_embd, head_size,bias=False)
        #Value
        self.value=nn.Linear(n_embd, head_size,bias=False)
        #Register buffer itu buat ngebuat tensor yang ga bakal di update sama optimizer
        #Berguna untuk tensor yang perlu disimpen tapi ga perlu di update, contoh kek mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        #Lower triangular matrix (torch.tril) for masking, preventing current token talking to future token
        # [[1,0,0,0,0],
        #  [1,1,0,0,0],
        #  [1,1,1,0,0],
        #  [1,1,1,1,0],
        #  [1,1,1,1,1]]

    def forward(self, x):
        B, T, C = x.shape
        #Query and Key matrix
        k=self.key(x) # (B,T,C)
        q=self.query(x) # (B,T,C)
        #Compute attention
        #Attention = softmax(QK^T/sqrt(d_k))V
        wei= q @ k.transpose(-2,-1) * C**(-0.5) # (B,T,C) @ (B,C,T) ---> (B,T,T)
        wei= wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) #Masking, prevent current token talking to future token (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        # perform weight aggregation of values
        v=self.value(x) # (B,T,C) #Value matrix
        out= wei@v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    # Multi-head attention itu gabungan dari beberapa head self-attention yang dilakukan secara paralel   
    # Multi-head attention =  Concat(head1, head2, ..., headN)W^O
    # Dimana headi= Attention (QW^Q_i, KW^K_i, VW^V_i)
    def __init__(self, num_heads, head_size):
        super().__init__()
        #Buat beberapa head self-attention sebanyak num_heads
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # nn.ModuleList: Container untuk menyimpan daftar sub-modul yang merupakan instance dari nn.Module. Perlu mengatur sendiri bagaimana sub-modul ini digunakan dalam metode forward. (Mirip kayak sequential yang udah pasti forwardnya berurutan, kalau ini di setting sendiri)
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1) #Concatenate hasil dari setiap head
    
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # setiap token secara langsung baca logits buat next token dari lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #Kemudian selain encoding table, tambahin encoding untuk posisi karakternya
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #self-attention head
        self.sa_head=Head(n_embd)
        #Buat linear layer buat jadiin logit
        self.lm_head=nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape #Batch, Time (Time itu berapa banyak karakter yang ada di sequence)
        
        #idx di pass dan bakal baca di embedding table yang di init dan ambil baris yang sesuai dgn idx
        #idx =43 bakal baca baris ke 43 dari embedding table
        token_embd = self.token_embedding_table(idx) # (B,T,C)
        #Tambahin posisi karakternya
        pos_emb=self.position_embedding_table(torch.arange(T, device=device)) # (T,C) -> aranging dari 0 sampai T-1, kemudian di embedding 
        #Gabungin token embedding dan posisi embedding
        x=token_embd+pos_emb # (B,T,C)
        #Lakukan self attention head
        x=self.sa_head(x) # (B,T,C)
        #Masukin ke linear layer
        logits = self.lm_head(x) # (B,T,C)
        
        
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
            # Potong idx ke last block_size token karena kita udah punya position embedding yang ukurannya cuman segede block_size, kalau lebih nanti out of size
            idx_cond = idx[:, -block_size:] # (B, T)
            # forward pass (Prediction mode)
            logits, loss = self(idx_cond)#Call object dirinya sendiri (forward)
            # ambil idx paling belakang dari T, karena ini adalah prediksi untuk token selanjutnya
            logits = logits[:, -1, :] # becomes (B, C)
            # pasang softmax untuk mendapatkan distribusi probabilitas
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample prediksi dari distribusi probabilitas
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # pasang idx_next yang di prediksi ke sequence yang udah ada buat ngulang lg
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
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