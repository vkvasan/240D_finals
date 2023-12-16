import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from my_utils import QuantEmbedding,QuantAct,IntSoftmax,LayerNorm,IntLayerNorm,QuantLinear

# hyperparameters
batch_size = 64 
block_size = 256 
max_iters = 5000  
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
head_size = int( n_embd/n_head )

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
print(chars)
vocab_size = len(chars)
print(vocab_size)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size,q_proj, k_proj, v_proj):
        super().__init__()
        self.fc_weight_bit = 8
        self.fc_bias_bit = 32
        self.quant_mode = 'symmetric'

        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.act_bit = 8
        self.softmax_output_bit=8
        self.force_dequant=None
        self.dropout = nn.Dropout(dropout)
        self.k_proj_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)    
        self.v_proj_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)     
        self.q_proj_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)     
        self.softmax = IntSoftmax(self.softmax_output_bit, quant_mode=self.quant_mode, force_dequant=self.force_dequant)

        self.key   = k_proj
        self.query = q_proj
        self.value = v_proj

    def forward(self, q,k,v,q_scaling_factor,k_scaling_factor,v_scaling_factor):
        B,T,Crhyhbv = q.shape
        q, q_scaling_factor = self.q_proj_act(q, q_scaling_factor)  
        k, k_scaling_factor = self.k_proj_act(k, k_scaling_factor)  
        v, v_scaling_factor = self.v_proj_act(v, v_scaling_factor)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        wei_scaling_factor=k_scaling_factor*q_scaling_factor
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei,wei_scaling_factor = self.softmax(wei, wei_scaling_factor)
        wei = self.dropout(wei)
        
        out = wei @ v 
        return out*wei_scaling_factor ,wei_scaling_factor

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        
        
        self.fc_weight_bit = 8
        self.fc_bias_bit = 32
        self.quant_mode = 'symmetric'

        proj = QuantLinear(self.fc_weight_bit, bias_bit=self.fc_bias_bit,     
                             quant_mode=self.quant_mode, per_channel=False)
        proj.set_param(nn.Linear(head_size * num_heads, n_embd, bias=True))         
        self.proj = proj

        self.dropout = nn.Dropout(dropout)

        
        k_proj = QuantLinear(self.fc_weight_bit, bias_bit=self.fc_bias_bit,     
                             quant_mode=self.quant_mode, per_channel=False)
        v_proj = QuantLinear(self.fc_weight_bit, bias_bit=self.fc_bias_bit,     
                             quant_mode=self.quant_mode, per_channel=False)
        q_proj = QuantLinear(self.fc_weight_bit, bias_bit=self.fc_bias_bit,      
                             quant_mode=self.quant_mode, per_channel=False)
        k_proj.set_param(nn.Linear(n_embd, n_embd, bias=True))        
        v_proj.set_param(nn.Linear(n_embd, n_embd, bias=True))        
        q_proj.set_param(nn.Linear(n_embd, n_embd, bias=True))         
        self.key   = k_proj
        self.query = q_proj
        self.value = v_proj

        self.heads = nn.ModuleList([Head(head_size,self.query, self.key,self.value) for _ in range(num_heads)])
         
         
    def forward(self, x,x_scaling_factor):
        

        k,k_scaling_factor = self.key(x,x_scaling_factor)  
        q,q_scaling_factor = self.query(x,x_scaling_factor)
        v,v_scaling_factor = self.value(x,x_scaling_factor)
        
        k_pkt = torch.chunk(k, chunks=n_head, dim=2)
        q_pkt = torch.chunk(q, chunks=n_head, dim=2)
        v_pkt = torch.chunk(v, chunks=n_head, dim=2)

        net_out = []
        net_scaling_factor = []
        for i,h in enumerate(self.heads):
            out1, out_scaling_factor = h(q_pkt[i],k_pkt[i],v_pkt[i],q_scaling_factor,k_scaling_factor,v_scaling_factor)
            net_out.append(out1)
            net_scaling_factor.append(out_scaling_factor)

        out = torch.cat( net_out , dim=-1)
        net_scaling_factor = torch.tensor( np.mean(net_scaling_factor) )
        overall_mean = net_scaling_factor.mean()
        overall_mean = overall_mean.unsqueeze(0)
        

        
        overall_mean = overall_mean.to(torch.float)
        out = out.to(torch.float)
        overall_mean_new = overall_mean.to(device)

        out,out_scaling_factor = self.proj(out,overall_mean_new)
        out = self.dropout(out)
        return out,out_scaling_factor

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x,x_scaling_factor):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.act_bit = 8
        self.quant_mode = 'symmetric'
        self.ln_output_bit = 32
        self.ln_bit = 22
        self.fc_weight_bit = 8
        self.fc_bias_bit = 32

        # layer norm associated with the self attention layer
        self_attn_layer_norm = nn.LayerNorm(n_embd)
        self.self_attn_layer_norm = IntLayerNorm(self.ln_output_bit, 
                                                 quant_mode='symmetric',
                                                 force_dequant = 'layernorm')
        self.self_attn_layer_norm.set_param(self_attn_layer_norm)


        # layer norm associated with the position wise feed-forward NN
        final_layer_norm = nn.LayerNorm(n_embd)
        self.final_layer_norm = IntLayerNorm(self.ln_output_bit, 
                                             quant_mode=self.quant_mode,
                                             force_dequant='layernorm')
        self.final_layer_norm.set_param(final_layer_norm)


        self.input_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.pre_self_attn_layer_norm_act = QuantAct(self.ln_bit, quant_mode=self.quant_mode)
        self.pre_final_layer_norm_act = QuantAct(self.ln_bit, quant_mode=self.quant_mode)
    def forward(self, x,x_scaling_factor):

        x, x_scaling_factor = self.input_act(x, x_scaling_factor)
        residual, residual_scaling_factor = x, x_scaling_factor
        
        x,x_scaling_factor  = self.sa(x,x_scaling_factor)
        x,x_scaling_factor = self.pre_self_attn_layer_norm_act(
                x, x_scaling_factor)
        x = x + residual 

        # LN1
        x, x_scaling_factor = self.self_attn_layer_norm(
                x, x_scaling_factor, self.quant_mode)
        
        
        x = x + self.ffwd(x,x_scaling_factor)
        # Pre LN2 activation (+ residual addition)
        x, x_scaling_factor = self.pre_final_layer_norm_act(
                x, x_scaling_factor,
                identity=residual,
                identity_scaling_factor=residual_scaling_factor)

        # LN2
        x, x_scaling_factor = self.final_layer_norm(x, x_scaling_factor)
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.embed_bit = 8
        self.quant_mode = 'symmetric'
        
        self.token_embedding_table = QuantEmbedding(weight_bit=self.embed_bit, quant_mode=self.quant_mode)
        self.token_embedding_table.set_param(nn.Embedding(vocab_size, n_embd))        
        self.position_embedding_table = QuantEmbedding(weight_bit=self.embed_bit, quant_mode=self.quant_mode, is_positional=True)
        self.position_embedding_table.set_param(nn.Embedding(block_size, n_embd))
        self.embed_act_bit = 8
        self.embed_positions_act = QuantAct(self.embed_act_bit, quant_mode=self.quant_mode)
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        x,x_scaling_factor = self.token_embedding_table(idx) # (B,T,C)

        y, y_scaling_factor = self.position_embedding_table(idx) 
        x, x_scaling_factor = self.embed_positions_act(x, x_scaling_factor,
                    identity=y,
                    identity_scaling_factor=y_scaling_factor
            )
        for block in self.blocks:
            x = block(x, x_scaling_factor)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
torch.save(model, 'trained_model')
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
