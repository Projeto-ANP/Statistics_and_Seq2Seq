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


if __name__ == "__main__":
    asyncio.run(main())
