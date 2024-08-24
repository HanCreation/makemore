import torch
import torch.nn as nn
import torch.nn.functional as F

#Hyperparameters
batch_size = 128
block_size= 256 #Context size to predict next character (context=8 berarti predict karakter ke 9 berdasarkan 8 urutan sebelumnya)
max_iters=10000
eval_interval=250 #Do evaluation every eval_interval iterations
learning_rate= 3e-4 #Self-attention ga terlalu bagus dengan learning rate yang tinggi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Use GPU (NVIDA RTX for example) if available
eval_iters=200
n_embd = 384 #Embedding dimensionsize
n_layer= 6 #Number of layers
n_head=6 #Number of heads in multi-head attention
dropout=0.2 #20% dropout
#------

# torch.manual_seed(1337)

#Data ('tinyshakespeare.txt' is a tiny version of 'shakespeare.txt')
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
file_name='puisi.txt' # change to tinyshakespeare directory/file name
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

        #Dropout bekerja dengan cara mematikan sebagian node/neuron secara acak, sehingga node lainnya harus belajar untuk mengambil alih pekerjaan node yang dimatikan (Banyak kombinasi neuron neuron acak)
        #Ini membantu mencegah overfitting
        self.dropout=nn.Dropout(dropout) #For preventing some nodes communicating

        

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
        wei = self.dropout(wei) #Dropout
        # perform weight aggregation of values
        v=self.value(x) # (B,T,C) #Value matrix
        out= wei@v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    # Multi-head attention itu gabungan dari beberapa head self-attention yang dilakukan secara paralel   
    # Multi-head attention =  Concat(head1, head2, ..., headN)W^O
    # Dimana headi= Attention (QW^Q_i, KW^K_i, VW^V_i)
    # Disini akan menguntungkan performance karena token punya banyak hal yang dibicarain (With multiple heads, we have multiple channel of communication)
    def __init__(self, num_heads, head_size):
        super().__init__()
        #Buat beberapa head self-attention sebanyak num_heads
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # nn.ModuleList: Container untuk menyimpan daftar sub-modul yang merupakan instance dari nn.Module. Perlu mengatur sendiri bagaimana sub-modul ini digunakan dalam metode forward. (Mirip kayak sequential yang udah pasti forwardnya berurutan, kalau ini di setting sendiri)
        
        #Projection digunakan untuk menggabungkan hasil dari setiap head
        self.proj=nn.Linear(n_embd, n_embd) #Projection matrix
        
        #Dropout layer
        self.dropout=nn.Dropout(dropout)
    
    def forward(self, x):
        out= torch.cat([h(x) for h in self.heads], dim=-1) #Concatenate hasil dari setiap head
        out = self.proj(out) #Projection
        out = self.dropout(out) #Dropout
        return out
    
class FeedForward(nn.Module):
    # Simple feedforward network with linear and non-linear activation    
    def __init__(self, n_embd):
        super().__init__()
        self.net=nn.Sequential( #Per token level
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd), #Projection layer
            nn.Dropout(dropout) #Dropout layer
        )
        
    def forward(self,x):
        out=self.net(x)
        return out
    
class Block(nn.Module):
    '''Transformer block: Attention dilanjutkan dengan komputasi'''
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size= n_embd//n_head
        #Attention -> Feedforward
        self.sa_head=MultiHeadAttention(n_head, head_size)
        self.ffwd=FeedForward(n_embd)
        
        #Pre norm formulation -> gak sama kayak paper original (Normalization Layer)
        #Layer norm is normalizing the features, unit gaussian at init
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)
        
    def forward(self,x):
        # x + f(x) -> Residual connection
        x=x + self.sa_head(self.ln1(x))
        x=x + self.ffwd(self.ln2(x))
        return x
    


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        
        # setiap token secara langsung baca logits buat next token dari lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #Kemudian selain encoding table, tambahin encoding untuk posisi karakternya
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        #self-attention head 1 head doang
        # self.sa_head=Head(n_embd)
        
        # multi-head attention 4 head -> Awalnya 1 head dengan 32 dimensi, sekarang 4 head dengan 8 dimensi yang digabung jadi 32 dimensi ujungnya
        # self.sa_heads=MultiHeadAttention(4, n_embd//4) #4 head, masing-masing head punya ukuran n_embd//4
        # self.ffwd=FeedForward(n_embd)
        
        #Multi block 
        # self.blocks=nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd) #Last layer norm sebelum masuk ke linear layer terakhir
        # )
        self.blocks=nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) #Buat n-layer
        self.ln=nn.LayerNorm(n_embd) #Last layer norm sebelum masuk ke linear layer terakhir
        
        # mulai deep neural netnya sehingga mulai ada isu optimasi -> fix dengan residual connection
        # Residual connection itu nambahin input ke outputnya, jadi outputnya = input + f(input), ini membantu gradient flow saat backpropagation
        
        #Buat linear layer buat jadiin logit -> Final layer
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
       
        # #Lakukan self attention head/multi-head
        # x=self.sa_heads(x) # (B,T,C)
        # # Feed-forward transformers
        # x=self.ffwd(x)
        
        #Multi block
        x=self.blocks(x) # (B,T,C)
        #Layer norm
        x=self.ln(x) # (B,T,C)
        
        #Masukin ke linear layer terakhir
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
        # idx adalah (B x T) tensor dari konteks saat ini
        for _ in range(max_new_tokens):
            # Potong idx ke last block_size token karena kita udah punya position embedding yang ukurannya cuman segede block_size, kalau lebih nanti out of size
            idx_cond = idx[:, -block_size:] # (B, T)
            # forward pass (Prediction mode)
            logits, loss = self(idx_cond)  # Call object dirinya sendiri (forward)
            # ambil idx paling belakang dari T, karena ini adalah prediksi untuk token selanjutnya
            logits = logits[:, -1, :]  # becomes (B, C)
            # pasang softmax untuk mendapatkan distribusi probabilitas
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample prediksi dari distribusi probabilitas
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # pasang idx_next yang di prediksi ke sequence yang udah ada buat ngulang lg
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            
            # Decode and print the latest token
            latest_char = decode([idx_next.item()])
            print(latest_char, end='', flush=True)  # Print without newline and flush immediately to output

        return idx


#load model
model_path='D:\Repository\makemore\ModelPuisi\model.pth'
model_load=GPTLanguageModel(vocab_size)
model_load.load_state_dict(torch.load(model_path))
m=model_load.to(device)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) #Start index
#  [0] itu First dimension/firstbatch for predicition -> m.generate(context, max_new_tokens=500)[0].tolist()
m.generate(context, max_new_tokens=1000)