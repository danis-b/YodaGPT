# inspired by original code from https://github.com/karpathy/nanoGPT
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna

# Fixed Hyperparameters
batch_size = 32
max_iters = 10000
eval_interval = 100
eval_iters = 200
n_head = 4
n_layer = 3
patience = 5
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

def get_batch(split, block_size):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, block_size):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class GPTDecoder(nn.Module):
    def __init__(self, n_embd, dropout, block_size):
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

    def generate(self, idx, max_new_tokens, temperature, top_k):
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

def objective(trial):
    # Suggest hyperparameters with Trial 99 as reference
    learning_rate = trial.suggest_float("learning_rate", 5e-4, 3e-3, log=True)  # Around 1.84e-3
    n_embd = trial.suggest_int("n_embd", 48, 96, step=8)  # 48, 56, 64, 72, 80, 88, 96
    dropout = trial.suggest_float("dropout", 0.2, 0.5)  # Around 0.42
    weight_decay = trial.suggest_float("weight_decay", 5e-2, 3e-1, log=True)  # Around 0.19
    warmup_steps = trial.suggest_int("warmup_steps", 400, 1000, step=100)  # Around 700
    top_k = trial.suggest_int("top_k", 2, 6)  # Around 3
    block_size = trial.suggest_int("block_size", 24, 96, step=8)  # 24, 32, 40, 48, 56, 64, 72, 80, 88, 96
    temperature = trial.suggest_float("temperature", 0.3, 1.0)  # New: Around 0.5

    # Suggest initial values from previous setup
    trial.set_user_attr("initial_learning_rate", 0.0024595055525029963)
    trial.set_user_attr("initial_n_embd", 96)
    trial.set_user_attr("initial_dropout", 0.43652873589534913)
    trial.set_user_attr("initial_weight_decay",  0.17511875633680343)
    trial.set_user_attr("initial_warmup_steps", 700)
    trial.set_user_attr("initial_top_k", 3)
    trial.set_user_attr("initial_block_size", 64)
    trial.set_user_attr("initial_temperature", 0.5)

    # Model setup
    model = GPTDecoder(n_embd=n_embd, dropout=dropout, block_size=block_size)
    m = model.to(device)
    print(f"Trial parameters: lr={learning_rate}, n_embd={n_embd}, dropout={dropout}, weight_decay={weight_decay}, warmup_steps={warmup_steps}, top_k={top_k}, block_size={block_size}, temperature={temperature}")

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

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(m, block_size)
            val_loss = losses['val'].item()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {val_loss:.4f}")

            improvement = best_val_loss - val_loss
            if improvement > early_stopping_threshold:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at iteration {iter}. Best val loss: {best_val_loss:.4f}")
                break

        xb, yb = get_batch('train', block_size)
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    return best_val_loss

# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=150)  # Increased for temperature inclusion

# Print best parameters and result
print("Best trial:")
trial = study.best_trial
print(f"  Value (val loss): {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Train final model with best parameters
best_params = study.best_params
model = GPTDecoder(n_embd=best_params["n_embd"], dropout=best_params["dropout"], block_size=best_params["block_size"])
m = model.to(device)
optimizer = torch.optim.AdamW(
    m.parameters(),
    lr=best_params["learning_rate"],
    weight_decay=best_params["weight_decay"],
    betas=(beta1, beta2),
)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(1.0, step / best_params["warmup_steps"]) * (0.75 + 0.25 * (1.0 - step / max_iters))
)

best_val_loss = float('inf')
patience_counter = 0
best_model_path = "best_net.pth"

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(m, best_params["block_size"])
        train_loss = losses['train']
        val_loss = losses['val']
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

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

    xb, yb = get_batch('train', best_params["block_size"])
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
print(decode(m.generate(context, max_new_tokens=200, temperature=best_params["temperature"], top_k=best_params["top_k"])[0].tolist()))