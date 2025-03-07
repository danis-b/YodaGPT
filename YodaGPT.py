# inspired by original nanoGPT code https://github.com/karpathy/nanoGPT
# for nanoGPT, please, find the license https://github.com/karpathy/nanoGPT?tab=MIT-1-ov-file

import torch
import torch.nn as nn
import torch.nn.functional as F

# Best Parameters from Optuna optimization
learning_rate = 0.002259809089341457
n_embd = 96
dropout = 0.21658394415968543
weight_decay = 0.057055997635279965
warmup_steps = 500
top_k =  2
block_size = 80
temperature = 0.8987989031765724

# Fixed Hyperparameters
batch_size = 64
max_iters = 10000
eval_interval = 100
eval_iters = 200
n_head = 4
n_layer = 3
patience = 10
early_stopping_threshold = 5e-3
beta1 = 0.9
beta2 = 0.95

# Device setup
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device is {device}")

torch.manual_seed(1337)

# Data loading
with open("quotes.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class GPTDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        x = self.transformer(x, mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = top_k_indices.gather(-1, idx_next)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Training setup
model = GPTDecoder()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

optimizer = torch.optim.AdamW(
    m.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=(beta1, beta2),
)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(1.0, step / warmup_steps) * (0.75 + 0.25 * (1.0 - step / max_iters))
)

best_val_loss = float('inf')
patience_counter = 0
best_model_path = "best_net.pth"

train_loss_plot = []
val_loss_plot = []

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(m)
        train_loss = losses['train']
        val_loss = losses['val']
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        train_loss_plot.append(train_loss.item())
        val_loss_plot.append(val_loss.item())

        improvement = best_val_loss - val_loss
        if improvement > early_stopping_threshold:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(m.state_dict(), best_model_path)
            print(f"New best model saved with val loss {best_val_loss:.4f} (improvement: {improvement:.4f})")
        else:
            patience_counter += 1
            print(f"No significant improvement ({improvement:.4f} < {early_stopping_threshold}). "
                  f"Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at iteration {iter}. Best val loss: {best_val_loss:.4f}")
            break

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

# Save final model
torch.save(m.state_dict(), 'net.pth')

# Load best model for generation
m.load_state_dict(torch.load(best_model_path))
m.eval()

# Generate with best model
context = torch.tensor([encode("Fear is the path to the dark")], dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))


with open('losses.txt', 'w') as out:
    for i, (train_loss, val_loss) in enumerate(zip(train_loss_plot, val_loss_plot)):
        print(i, train_loss, val_loss, file=out)
        
    
    
