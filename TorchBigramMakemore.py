import torch
import torch.nn as nn
import torch.nn.functional as F

#Hyperparameters
batch_size = 32
block_size=8
max_iters=3000
eval_interval=300 #Do evaluation every eval_interval iterations
lr= 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
print("".join(chars))
print(vocab_size)
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
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])# targetnya adalah x yang di offset 1
    #torch.stack itu buat numpuk tensor-tensor 1D terus di tumpuk semua (stack them up at rows)
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")