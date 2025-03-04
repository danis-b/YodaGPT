import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Define hyperparameters (consistent with your model)
batch_size = 32
block_size = 64
n_embd = 48
n_head = 4
n_layer = 3
dropout = 0.3
vocab_size = None  # Will be set after loading data

# Device setup
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load vocabulary and data utilities
with open("quotes.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]  # Handle unknown chars gracefully
decode = lambda l: "".join([itos[i] for i in l])

# Define the GPTDecoder class (same as your code)
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
        mask = mask.masked_fill(mask == 1, float("-inf"))
        x = self.transformer(x, mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.5, top_k=3):
        generated = idx
        for _ in range(max_new_tokens):
            idx_cond = generated[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = top_k_indices.gather(-1, idx_next)
            generated = torch.cat((generated, idx_next), dim=1)
            yield generated  # Yield the current sequence at each step
        return generated

# Load the model
model = GPTDecoder()
model.load_state_dict(torch.load("yoda_weights.pth", map_location=device))
model.to(device)
model.eval()

# Streamlit interface
st.set_page_config(layout="wide")

# Initialize session state for generation control and text storage
if "generating" not in st.session_state:
    st.session_state.generating = False
if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""

# Sidebar for parameters
st.sidebar.image("logo.png")
st.sidebar.title("Generation Parameters")
max_new_tokens = st.sidebar.slider(
    "Max new tokens",
    min_value=50,
    max_value=500,
    value=200,
    step=10,
    help="Number of new characters to generate",
)
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1,
    help="Controls randomness: lower (0.3) for predictable text, higher (1.2) for creative text.",
)
top_k = st.sidebar.slider(
    "Top-K",
    min_value=1,
    max_value=10,
    value=2,
    step=1,
    help="Limits choices to top K options: lower (2) for focused text, higher (6) for more variety.",
)

# Main content area
st.title("YodaGPT text generator")
st.write(
    "Enter a prompt below and click the Generate/Stop button to start/stop the generation of text in Yoda's style."
)

# Prompt input
prompt = st.text_area("Prompt", value="Fear is the path to the", height=100)

# Placeholder for output
output_placeholder = st.empty()

# Display the initial text (prompt + previously generated text if any)
output_placeholder.text(prompt + st.session_state.generated_text)

# Generate/Stop button logic
button_label = "Generate/Stop"
if st.button(button_label):
    if not st.session_state.generating:
        # Start generation
        st.session_state.generating = True
        st.session_state.generated_text = ""  # Reset only when starting anew
        with st.spinner("Generating text..."):
            # Encode the prompt
            context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            initial_text = prompt
            output_placeholder.text(initial_text)
            word_buffer = ""

            for generated_idx in model.generate(
                context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            ):
                if not st.session_state.generating:
                    break  # Stop generation if toggled off

                new_char = itos[generated_idx[0, -1].item()]
                word_buffer += new_char

                # Check if we've hit a space or punctuation to signify a word boundary
                if new_char in " .,!?;:\n" and word_buffer.strip():
                    initial_text += word_buffer
                    st.session_state.generated_text += word_buffer
                    output_placeholder.text(initial_text)
                    word_buffer = ""
                    time.sleep(0.3)  # Add a small delay for visual effect
                elif new_char not in " .,!?;:\n":
                    continue  # Keep buffering until a word is complete

            # Flush any remaining buffer
            if word_buffer.strip() and st.session_state.generating:
                initial_text += word_buffer
                st.session_state.generated_text += word_buffer
                output_placeholder.text(initial_text)

            # Reset generating state when done
            st.session_state.generating = False
    else:
        # Stop generation
        st.session_state.generating = False
        # Keep the generated text in output_placeholder
        output_placeholder.text(prompt + st.session_state.generated_text)
else:
    if not st.session_state.generating:
        st.write("Click 'Generate/Stop' to start generation.")
    # Ensure the text persists even when not generating
    output_placeholder.text(prompt + st.session_state.generated_text)
    
    

# Add GitHub link at the bottom
st.markdown(
    """
    <div style='text-align: center; position: fixed; bottom: 10px; width: 100%;'>
        <a href='https://github.com/danis-b/YodaGPT' target='_blank'>View this project on GitHub (danis-b)</a>
    </div>
    """,
    unsafe_allow_html=True
)