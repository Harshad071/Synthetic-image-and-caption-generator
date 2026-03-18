import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pickle
import math
import io

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Image Captioning AI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# DEVICE SETUP
# =====================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================
# TOKENIZER CLASS
# =====================================
class Tokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word_freq = {}

    def build_vocab(self, captions):
        for caption in captions:
            words = caption.lower().split()
            for word in words:
                self.word_freq[word] = self.word_freq.get(word, 0) + 1

        idx = 4
        for word, _ in sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size - 4]:
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

    def encode(self, caption, max_len=50):
        tokens = [self.word2idx['<SOS>']]
        words = caption.lower().split()
        for word in words:
            if len(tokens) >= max_len:
                break
            tokens.append(self.word2idx.get(word, self.word2idx['<UNK>']))
        tokens.append(self.word2idx['<EOS>'])
        return tokens

    def decode(self, indices):
        words = []
        for idx in indices:
            if idx == self.word2idx['<EOS>']:
                break
            if idx != self.word2idx['<SOS>'] and idx != self.word2idx['<PAD>']:
                words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)

# =====================================
# MODEL ARCHITECTURE
# =====================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embed_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, captions, encoder_output, tgt_mask=None):
        x = self.embedding(captions)
        x = self.pos_encoding(x)
        memory = encoder_output.unsqueeze(1).expand(-1, x.size(1), -1)
        x = self.transformer_decoder(x, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(x)
        return logits

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512):
        super().__init__()
        self.encoder = CNNEncoder(embed_dim)
        self.decoder = TransformerDecoder(vocab_size, embed_dim)

    def forward(self, images, captions, tgt_mask=None):
        enc_out = self.encoder(images)
        logits = self.decoder(captions, enc_out, tgt_mask)
        return logits

# =====================================
# CACHING FUNCTIONS
# =====================================
@st.cache_resource
def load_tokenizer():
    """Load tokenizer from file"""
    try:
        with open('tokenizer (3).pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Tokenizer file 'tokenizer (3).pkl' not found!")
        return None
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        return None

@st.cache_resource
def load_model(vocab_size, model_path='model_epoch_50.pt'):
    """Load model from file"""
    try:
        # Check if model file exists
        try:
            with open(model_path, 'rb'):
                pass
        except FileNotFoundError:
            st.error(f"Model file '{model_path}' not found!")
            return None
        
        model = ImageCaptioningModel(vocab_size=vocab_size).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# =====================================
# IMAGE PREPROCESSING
# =====================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =====================================
# MASK CREATION
# =====================================
def create_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    return mask.to(device)

# =====================================
# CAPTION GENERATION
# =====================================
def generate_caption(model, image_tensor, tokenizer, top_k=5, temperature=1.0):
    """Generate caption for image using top-k sampling with temperature"""
    with torch.no_grad():
        encoder_output = model.encoder(image_tensor)

    caption_tokens = [tokenizer.word2idx['<SOS>']]

    while True:
        input_tokens = torch.tensor([caption_tokens], dtype=torch.long).to(device)
        tgt_mask = create_causal_mask(input_tokens.size(1))

        with torch.no_grad():
            logits = model.decoder(input_tokens, encoder_output, tgt_mask=tgt_mask)

        # Apply temperature scaling
        logits = logits[0, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_idx = probs.topk(top_k)
        next_token = top_k_idx[torch.multinomial(top_k_probs, 1)].item()
        caption_tokens.append(next_token)

        if next_token == tokenizer.word2idx['<EOS>'] or len(caption_tokens) > 100:
            break

    caption = tokenizer.decode(caption_tokens)
    return caption

# =====================================
# STREAMLIT UI
# =====================================
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%); }
    .stTitle { color: #00d4ff; font-size: 2.5em !important; }
    .caption-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 20px 0;
        font-size: 1.1em;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("🎨 Image Captioning AI")
st.markdown("*Generate intelligent descriptions for your images using CNN + Transformer*")

# =====================================
# LOAD MODEL AND TOKENIZER
# =====================================
try:
    with st.spinner("Loading model and tokenizer..."):
        tokenizer = load_tokenizer()
        
    if tokenizer is None:
        st.stop()
        
    # Model selection
    st.subheader("🤖 Model Selection")
    col1, col2 = st.columns([3, 1])
    with col1:
        model_option = st.selectbox(
            "Select Model",
            ["model_epoch_50.pt (Best)", "model_epoch_25.pt"],
            help="Choose between model_epoch_50 (best quality) or model_epoch_25 (faster)",
            key="model_select"
        )
    with col2:
        st.write("")
        st.write("")
        reload = st.button("🔄 Reload Model", help="Click to load selected model")
    
    model_path = "model_epoch_50.pt" if "50" in model_option else "model_epoch_25.pt"
    
    # Try to load the selected model (use session_state to cache)
    if 'current_model' not in st.session_state or st.session_state.get('current_model_path') != model_path or reload:
        with st.spinner(f"Loading {model_option}..."):
            model = load_model(vocab_size=len(tokenizer.word2idx), model_path=model_path)
        if model is not None:
            st.session_state['current_model'] = model
            st.session_state['current_model_path'] = model_path
    
    model = st.session_state.get('current_model')
    
    if model is None:
        st.stop()

    st.success("✅ Model loaded successfully!")
    st.info(f"Device: {device} | GPU: {'Available ✅' if torch.cuda.is_available() else 'Not Available ❌'}")

    # =====================================
    # MAIN: IMAGE UPLOAD & GENERATION
    # =====================================
    st.subheader("📷 Upload Image")
    image_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if image_file:
        # Display image
        col1, col2 = st.columns([1, 1])

        with col1:
            image = Image.open(image_file).convert('RGB')
            st.image(image, use_column_width=True, caption="Uploaded Image")

        with col2:
            # Caption generation parameters
            st.subheader("⚙️ Generation Settings")
            top_k = st.slider("Top-K Sampling", min_value=1, max_value=20, value=5)
            temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1, help="Higher = more creative, Lower = more deterministic")

            if st.button("🚀 Generate Caption", use_container_width=True):
                with st.spinner("Generating caption..."):
                    # Preprocess image
                    image_tensor = transform(image).unsqueeze(0).to(device)

                    # Generate caption
                    caption = generate_caption(model, image_tensor, tokenizer, top_k, temperature)

                    # Display result
                    st.markdown(f"""
                    <div class="caption-box">
                        <strong>📝 Generated Caption:</strong><br>
                        {caption.capitalize()}
                    </div>
                    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.info("Ensure 'tokenizer (3).pkl' and 'model_epoch_50.pt' are in the same directory as the app.")
