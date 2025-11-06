import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd

# ======================================================
# 1. Configurações
# ======================================================
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # ou outro LLaMA compatível
SEQ_LEN = 24  # janela de tempo
PRED_HORIZON = 6  # passos à frente para prever
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# 2. Dataset sintético (exemplo)
# ======================================================
np.random.seed(42)
time = np.arange(200)
series = np.sin(time / 6) + np.random.normal(0, 0.1, len(time))
df = pd.DataFrame({"value": series})


def create_patches(data, seq_len, pred_horizon):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_horizon):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pred_horizon])
    return np.array(X), np.array(y)


X, y = create_patches(df["value"].values, SEQ_LEN, PRED_HORIZON)
X_train, y_train = torch.tensor(X[:150], dtype=torch.float32), torch.tensor(
    y[:150], dtype=torch.float32
)
X_test, y_test = torch.tensor(X[150:], dtype=torch.float32), torch.tensor(
    y[150:], dtype=torch.float32
)


# ======================================================
# 3. Quantização (opcional)
# ======================================================
def quantize(x, num_bins=64):
    min_val, max_val = x.min(), x.max()
    bins = torch.linspace(min_val, max_val, num_bins)
    quantized = torch.bucketize(x, bins)
    return quantized, bins


# ======================================================
# 4. Modelo: LLaMA congelado + cabeça de regressão
# ======================================================
class TimeLLM(nn.Module):
    def __init__(self, model_name, embed_dim=768, pred_horizon=6):
        super().__init__()
        # Tokenizer + modelo congelado
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        for p in self.llm.parameters():
            p.requires_grad = False

        # Camadas aprendíveis
        self.patch_embed = nn.Linear(SEQ_LEN, embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, pred_horizon)
        )

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.patch_embed(x)  # (batch, embed_dim)

        # Convertemos para formato textual "prompt-like"
        prompts = [
            " ".join([f"{v:.3f}" for v in seq]) for seq in x.detach().cpu().numpy()
        ]
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)

        # Passagem pelo LLaMA congelado
        with torch.no_grad():
            outputs = self.llm(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)  # pooling

        # Previsão via MLP
        out = self.regressor(features)
        return out


# ======================================================
# 5. Treinamento leve
# ======================================================
model = TimeLLM(MODEL_NAME, embed_dim=768, pred_horizon=PRED_HORIZON).to(DEVICE)
optimizer = torch.optim.Adam(model.regressor.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(5):  # poucos epochs bastam
    model.train()
    optimizer.zero_grad()
    pred = model(X_train.to(DEVICE))
    loss = criterion(pred, y_train.to(DEVICE))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.6f}")

# ======================================================
# 6. Avaliação
# ======================================================
model.eval()
with torch.no_grad():
    preds = model(X_test.to(DEVICE)).cpu().numpy()

mse = np.mean((preds - y_test.numpy()) ** 2)
print(f"\nMSE no conjunto de teste: {mse:.6f}")
