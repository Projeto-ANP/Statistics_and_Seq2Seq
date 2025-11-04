from agno.agent import Agent, RunOutput  # noqa
from agno.models.google import Gemini
import asyncio
import os
import pandas as pd
import re
from agno.db.json import JsonDb
from textwrap import dedent
from agno.models.cerebras import Cerebras


GOOGLE_API_KEY = "AIzaSyC6yMJ3yCihm9TDfyoydv2diqtEWuIKeeE"
CEREBRAS_API_KEY = "csk-45wwjdx4fwyrvvekmxrjd29hf6382w9rnwwk5rttvwt42hkr"


def extract_values(list_str):
    if isinstance(list_str, str):
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", list_str)
        return [float(num) for num in numbers]
    return []


def read_model_preds(model_name, dataset_index):
    df = pd.read_csv(
        f"./Statistics_and_Seq2Seq/timeseries/mestrado/resultados/{model_name}/normal/ANP_MONTHLY.csv",
        sep=";",
    )
    df = df[df["dataset_index"] == dataset_index]

    df["start_test"] = pd.to_datetime(df["start_test"], format="%Y-%m-%d")
    df["final_test"] = pd.to_datetime(df["final_test"], format="%Y-%m-%d")
    df = df.sort_values(by="start_test")

    return df


def train_split_agent(models, dataset_index=0, final_test="2024-11-30"):
    from datetime import datetime

    all_data = {}
    final_test_predictions = {}
    final_test_data = {}

    final_test_date = pd.to_datetime(final_test, format="%Y-%m-%d")

    for model in models:
        df = read_model_preds(model, dataset_index)

        train_df = df[df["final_test"] < final_test_date]
        train_df = train_df[train_df["start_test"] >= "2010-12-31"]
        test_df = df[df["final_test"] == final_test_date]
        all_data[model] = []

        for index, row in train_df.iterrows():
            preds_model = extract_values(row["predictions"])
            test = extract_values(row["test"])
            all_data[model].append(
                {"predictions": preds_model, "test": test, "date": row["start_test"]}
            )

        if not test_df.empty:
            final_row = test_df.iloc[0]
            final_test_predictions[model] = extract_values(final_row["predictions"])
            final_test_data[model] = extract_values(final_row["test"])

    output_filename = f"training_data_dataset_{dataset_index}-{final_test}.txt"

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("<train>\n")

        first_model = models[0]
        num_examples = len(all_data[first_model])

        for example_idx in range(num_examples):
            f.write("    <example>\n")
            f.write("        <predictions>\n")

            for model in models:
                if example_idx < len(all_data[model]):
                    preds = all_data[model][example_idx]["predictions"]
                    f.write(f"            {model}: {preds}\n")

            f.write("        </predictions>\n")

            test_values = all_data[first_model][example_idx]["test"]
            f.write(f"        <test>{test_values}</test>\n")

            f.write("    </example>\n")

        f.write("</train>\n")

    # calcular a combinacao de media de todos os modelos em final_test_predictions
    mean_predictions = []
    for i in range(len(final_test_predictions[models[0]])):
        mean_value = sum(
            final_test_predictions[model][i]
            for model in models
            if model in final_test_predictions
        ) / len([model for model in models if model in final_test_predictions])
        mean_predictions.append(mean_value)

    # Criar string com predições do final_test no formato <predictions>
    predictions_string = "<predictions>\n"
    for model in models:
        if model in final_test_predictions:
            predictions_string += (
                f"            {model}: {final_test_predictions[model]}\n"
            )
    predictions_string += "</predictions>"

    # Usar o test do primeiro modelo como referência (assumindo que todos têm o mesmo test real)
    test_values = final_test_data.get(models[0], []) if final_test_data else []

    # print(
    #     f"Arquivo '{output_filename}' gerado com {num_examples} exemplos de treinamento"
    # )
    # print(f"Predições para {final_test}: {len(final_test_predictions)} modelos")
    # print(f"Valores reais para {final_test}: {test_values}")

    return (
        output_filename,
        predictions_string,
        test_values,
        mean_predictions,
        num_examples,
    )


async def main():
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["CEREBRAS_API_KEY"] = CEREBRAS_API_KEY

    models = [
        "ARIMA",
        "ETS",
        "THETA",
        "svr",
        "rf",
        "catboost",
        "CWT_svr",
        "DWT_svr",
        "FT_svr",
        "CWT_rf",
        "DWT_rf",
        "FT_rf",
        "CWT_catboost",
        "DWT_catboost",
        "FT_catboost",
    ]

    output_filename, predictions_string, test_values, mean_predictions, num_examples = (
        train_split_agent(models, dataset_index=146, final_test="2024-11-30")
    )

    # print(test_values)
    # print(predictions_string)

    with open(output_filename, "r", encoding="utf-8") as f:
        training_data_content = f.read()

    # Criar o agente
    agent = Agent(
        # model=Gemini(id="gemini-2.5-pro", max_output_tokens=4096),
        model=Cerebras(
            id="qwen-3-coder-480b",
            temperature=0.7,
            top_p=0.6,
            max_completion_tokens=8192,
            extra_body={"seed": 42},
        ),
        markdown=True,
    )

    prompt = f"""
    **DADOS DE TREINAMENTO:**
    1. Você recebera como dado de treinamento múltiplos exemplos no formato <example> contendo as predições dos modelos e o valor real observado (test).
    2. São {num_examples} exemplos de treinamento no total.
    
    Faça uma analise de cada uma das séries temporais e me diga o calculo de cada uma dá media das predições
    Dados de treinamento:
    {training_data_content}

    """

    # await agent.aprint_response(prompt)
    response: RunOutput = agent.run(prompt)
    print(response.content)


import numpy as np
from streamfuels.datasets import DatasetLoader


def timellm():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import numpy as np
    from streamfuels.datasets import DatasetLoader

    # 1. CARREGAR DADOS
    tsf_file = "australian_electricity_demand_dataset"
    dataset = tsf_file.split(".")[0].upper()
    file_path = f"../forecasting_datasets/{tsf_file}.tsf"
    loader = DatasetLoader()
    df, metadata = loader.read_tsf(path_tsf=file_path)

    if metadata["horizon"] == None:
        if metadata["frequency"] == "hourly":
            metadata["horizon"] = 24
        elif metadata["frequency"] == "daily":
            metadata["horizon"] = 14
        elif metadata["frequency"] == "half_hourly":
            metadata["horizon"] = 48

    df["series_value"] = df["series_value"].apply(np.array)
    series = df.iloc[0]["series_value"]
    print(series)

    # Parâmetros
    seq_len = 512  # Tamanho do contexto
    pred_len = metadata["horizon"]  # Horizonte de previsão
    patch_len = 16  # Tamanho de cada patch
    stride = 8  # Stride para criar patches

    # =========================================================================
    # STEP 1: NORMALIZAÇÃO (Z-NORM)
    # =========================================================================
    def znorm(x):
        """Z-normalization (média 0, desvio 1)"""
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return x - mean, mean, 1e-5
        return (x - mean) / std, mean, std

    # Pegar últimos seq_len valores
    if len(series) > seq_len:
        x_raw = series[-seq_len:]
    else:
        x_raw = series

    # Normalizar
    x_norm, mean_val, std_val = znorm(x_raw)
    x_norm_tensor = torch.tensor(x_norm).float()

    print("=" * 80)
    print("STEP 1: NORMALIZAÇÃO (Z-NORM)")
    print(f"Série original (shape): {x_raw.shape}")
    print(f"Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print(f"Série normalizada (primeiros 10): {x_norm[:10]}")

    # =========================================================================
    # STEP 2: CRIAR PATCHES EM JANELAS DO HORIZON
    # =========================================================================
    def create_patches(x, patch_len, stride):
        """Cria patches com stride"""
        patches = []
        for i in range(0, len(x) - patch_len + 1, stride):
            patch = x[i : i + patch_len]
            patches.append(patch)
        return torch.stack(patches) if patches else torch.tensor([])

    patches = create_patches(x_norm_tensor, patch_len, stride)
    num_patches = patches.shape[0]

    print("\n" + "=" * 80)
    print("STEP 2: CRIAR PATCHES")
    print(f"Patch length: {patch_len}, Stride: {stride}")
    print(f"Número de patches: {num_patches}")
    print(f"Shape dos patches: {patches.shape}")  # [num_patches, patch_len]
    print(f"Primeiro patch: {patches[0]}")

    # =========================================================================
    # STEP 3: EMBEDDINGS DOS PATCHES
    # =========================================================================
    def patch_embedding(patches, d_model=768):
        """
        Cria embeddings dos patches usando projeção linear
        Similar ao patch embedding em Vision Transformers
        """
        # Projeção linear: [num_patches, patch_len] -> [num_patches, d_model]
        projection = torch.nn.Linear(patch_len, d_model)
        embeddings = projection(patches)

        # Adicionar positional encoding
        num_patches = embeddings.shape[0]
        position = torch.arange(num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pos_encoding = torch.zeros(num_patches, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        embeddings = embeddings + pos_encoding

        return embeddings, projection

    embeddings, projection_layer = patch_embedding(patches)

    print("\n" + "=" * 80)
    print("STEP 3: EMBEDDINGS DOS PATCHES")
    print(f"Shape dos embeddings: {embeddings.shape}")  # [num_patches, d_model]
    print(f"Embedding do primeiro patch (primeiros 10 dims): {embeddings[0, :10]}")

    # =========================================================================
    # STEP 4: REPROGRAMAÇÃO DOS PATCHES (Patch Reprogramming)
    # =========================================================================
    def patch_reprogramming(embeddings, projection_layer):
        """
        Reprograma os patches para formato textual compreensível pela LLM
        Converte embeddings numéricos em tokens textuais
        """
        # Converter embeddings em representação textual estatística
        patch_descriptions = []

        for i, emb in enumerate(embeddings):
            # Estatísticas do embedding
            mean_emb = emb.mean().item()
            std_emb = emb.std().item()
            min_emb = emb.min().item()
            max_emb = emb.max().item()

            # Pegar o patch original correspondente
            patch_data = patches[i].numpy()

            # Criar descrição textual do patch
            patch_desc = (
                f"Patch {i+1}/{num_patches}: "
                f"[{', '.join([f'{v:.3f}' for v in patch_data[:5]])}...] "
                f"(mean={np.mean(patch_data):.3f}, "
                f"std={np.std(patch_data):.3f}, "
                f"trend={'↑' if patch_data[-1] > patch_data[0] else '↓'})"
            )
            patch_descriptions.append(patch_desc)

        # Criar representação compacta
        compact_repr = []
        for i in range(0, len(patch_descriptions), 5):
            batch = patch_descriptions[i : i + 5]
            compact_repr.append("\n".join(batch))

        return "\n\n".join(compact_repr), patch_descriptions

    reprogrammed_text, patch_descriptions = patch_reprogramming(
        embeddings, projection_layer
    )

    print("\n" + "=" * 80)
    print("STEP 4: REPROGRAMAÇÃO DOS PATCHES")
    print(f"Número de descrições: {len(patch_descriptions)}")
    print("\nPrimeiros 3 patches reprogramados:")
    for desc in patch_descriptions[:3]:
        print(f"  {desc}")

    # =========================================================================
    # STEP 5: CRIAR PROMPT PARA LLM
    # =========================================================================
    global_stats = {
        "mean": mean_val,
        "std": std_val,
        "min": x_raw.min(),
        "max": x_raw.max(),
        "trend": "increasing" if x_raw[-1] > x_raw[0] else "decreasing",
    }

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a time series forecasting expert. You analyze patch-based representations of time series data to make predictions.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Dataset: {dataset}
Frequency: {metadata["frequency"]}
Task: Predict next {pred_len} values

GLOBAL STATISTICS:
- Original Mean: {global_stats['mean']:.4f}
- Original Std: {global_stats['std']:.4f}
- Min: {global_stats['min']:.4f}
- Max: {global_stats['max']:.4f}
- Overall Trend: {global_stats['trend']}

PATCH-BASED REPRESENTATION:
(Each patch contains {patch_len} consecutive normalized time points)

{reprogrammed_text}

INSTRUCTIONS:
1. Analyze the patterns in these patches
2. Identify trends, seasonality, and anomalies
3. Generate {pred_len} future normalized values
4. Output format: comma-separated numbers only

Prediction:
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

    print("\n" + "=" * 80)
    print("STEP 5: PROMPT GERADO")
    print(f"Tamanho do prompt: {len(prompt)} caracteres")

    # =========================================================================
    # STEP 6: GERAR PREVISÃO COM LLM
    # =========================================================================
    if not torch.cuda.is_available():
        model_name = "google/flan-t5-large"  # Modelo menor
        print(f"GPU não disponível, usando modelo menor: {model_name}")

        from transformers import AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        print("Gerando previsão...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
            )
    else:
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        print(f"GPU disponível, usando: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("cuda")

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to("cuda")

        print("Gerando previsão...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

    print("Gerando previsão...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n\n\n TEXTO Gerado:")
    print(generated_text)
    prediction_text = generated_text.split("assistant<|end_header_id|>")[-1].strip()

    # =========================================================================
    # STEP 7: PROCESSAR SAÍDA E DESNORMALIZAR
    # =========================================================================
    import re

    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", prediction_text)
    predictions_norm = [float(num) for num in numbers[:pred_len]]

    # Desnormalizar
    predictions_denorm = [p * std_val + mean_val for p in predictions_norm]

    print("\n" + "=" * 80)
    print("STEP 7: RESULTADO FINAL")
    print(f"\nResposta da LLM:")
    print(prediction_text[:500])
    print(f"\nPrevisões normalizadas: {predictions_norm}")
    print(f"\nPrevisões desnormalizadas: {predictions_denorm}")

    return {
        "predictions_normalized": predictions_norm,
        "predictions": predictions_denorm,
        "mean": mean_val,
        "std": std_val,
        "num_patches": num_patches,
        "patch_len": patch_len,
    }


if __name__ == "__main__":
    result = timellm()
