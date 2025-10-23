from urllib import response
from agno.agent import Agent, RunOutput  # noqa
from agno.models.google import Gemini
import asyncio
import os
import pandas as pd
import re
from agno.db.json import JsonDb
from textwrap import dedent
from agno.models.cerebras import Cerebras
from agno.models.mistral import MistralChat
from pydantic import BaseModel, Field
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape


def pocid(y_true, y_pred):
    n = len(y_true)
    D = [
        1 if (y_pred[i] - y_pred[i - 1]) * (y_true[i] - y_true[i - 1]) > 0 else 0
        for i in range(1, n)
    ]
    POCID = 100 * np.sum(D) / (n - 1)
    return POCID


def calculate_smape(forecasts, test_set):
    smape = 2 * np.abs(forecasts - test_set) / (np.abs(forecasts) + np.abs(test_set))
    smape_per_series = np.nanmean(smape, axis=1)  # Média por série
    return smape_per_series


# Função para calcular os valores de mSMAPE
def calculate_msmape(forecasts, test_set):
    epsilon = 0.1
    comparator = np.full(test_set.shape, 0.5 + epsilon)
    sum_values = np.maximum(
        comparator, (np.abs(forecasts) + np.abs(test_set) + epsilon)
    )
    smape = 2 * np.abs(forecasts - test_set) / sum_values
    msmape_per_series = np.nanmean(smape, axis=1)
    return msmape_per_series


# Função para calcular os valores de MAE
def calculate_mae(forecasts, test_set):
    mae = np.abs(forecasts - test_set)
    mae_per_series = np.nanmean(mae, axis=1)
    return mae_per_series


# Função para calcular os valores de RMSE
def calculate_rmse(forecasts, test_set):
    squared_errors = (forecasts - test_set) ** 2
    rmse_per_series = np.sqrt(np.nanmean(squared_errors, axis=1))
    return rmse_per_series


class PredictionInformation(BaseModel):
    description: str = Field(
        ...,
        description="Models that were good at which points in the time series to generate the final prediction.",
    )
    result: str = Field(
        ...,
        description="Final combined predictions as a list of floats.",
    )


def extract_values(list_str):
    if isinstance(list_str, str):
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", list_str)
        return [float(num) for num in numbers]
    return []


def read_model_preds(model_name, dataset_index):
    df = pd.read_csv(
        # f"./Statistics_and_Seq2Seq/timeseries/mestrado/resultados/{model_name}/normal/ANP_MONTHLY.csv",
        f"./timeseries/mestrado/resultados/{model_name}/normal/ANP_MONTHLY.csv",
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
        train_df = train_df[train_df["start_test"] >= "2015-12-31"]
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
    GOOGLE_API_KEY = "AIzaSyC6yMJ3yCihm9TDfyoydv2diqtEWuIKeeE"
    # CEREBRAS_API_KEY = "csk-45wwjdx4fwyrvvekmxrjd29hf6382w9rnwwk5rttvwt42hkr"
    CEREBRAS_API_KEY = "csk_396yv6vjee9xkhce3xvyrtxdk4t649kyh48nyxtwk5x9ewdh"
    MISTRAL_API_KEY = (
        "VtBuj7crXH7dVVxXVlvj6l5m6lC5rHYn"  # Replace with your actual Mistral API key
    )

    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["CEREBRAS_API_KEY"] = CEREBRAS_API_KEY
    os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
    cols_serie = [
        "dataset_index",
        "horizon",
        "regressor",
        "mape",
        "pocid",
        "smape",
        "rmse",
        "msmape",
        "mae",
        "test",
        "predictions",
        "start_test",
        "final_test",
    ]
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

    # model_name = "mistral-small-2503"
    model_name = "qwen-3-coder-480b"
    exp_name = f"agent_{model_name}"
    path_experiments = f"./timeseries/mestrado/resultados/{exp_name}/"
    path_csv = f"{path_experiments}/ANP_MONTHLY.csv"
    os.makedirs(path_experiments, exist_ok=True)
    for dataset_index in range(71, 182):
        # dataset_index = 70
        final_test = "2024-11-30"

        (
            output_filename,
            predictions_string,
            test_values,
            mean_predictions,
            num_examples,
        ) = train_split_agent(
            models, dataset_index=dataset_index, final_test=final_test
        )

        # print(test_values)
        # print(predictions_string)

        with open(output_filename, "r", encoding="utf-8") as f:
            training_data_content = f.read()

        # Criar o agente
        agent = Agent(
            # model=Gemini(id="gemini-2.5-pro", max_output_tokens=4096),
            model=Cerebras(
                id=model_name,
                temperature=0.7,
                top_p=0.6,
                max_completion_tokens=4096,
                extra_body={"seed": 42},
            ),
            output_schema=PredictionInformation,
            # model=MistralChat(id=model_name, random_seed=42),
            markdown=True,
            description=dedent(
                """\
                Você é um agente especializado em combinar predições de múltiplos modelos de séries temporais para gerar uma predição mais precisa.
                Seu objetivo é analisar os padrões nas predições fornecidas por diferentes modelos e aprender a partir de exemplos anteriores como combinar essas predições para se aproximar dos valores reais observados.
                                
                Essa combinação será feita ponto a ponto, ou seja, você pegará os melhores modelos que são bons em cada ponto. Exemplo: ponto 1 o ARIMA sempre foi melhor, ponto 2 o RF sempre foi melhor, ponto 3 o CATBOOST sempre foi melhor. E assim por diante.
                """
            ),
            instructions=dedent(
                """\
                                Aprender, a partir de exemplos de predições de múltiplos modelos e seus valores reais observados, como combinar as predições desses modelos para gerar uma previsão mais próxima possível do valor real.

                                DADOS DE TREINAMENTO:
                                Você receberá múltiplos blocos de dados no formato <example>, cada um contendo:
                                - As predições de vários modelos para uma mesma série temporal.
                                - O valor real observado correspondente (<test>).

                                O QUE O AGENTE DEVE APRENDER:
                                1. Identificar quais modelos tendem a se aproximar mais do valor real (<test>) em diferentes contextos e padrões das séries.
                                2. Aprender uma lógica de combinação adaptativa, que pode envolver:
                                - Dar maior peso aos modelos historicamente mais precisos;
                                - Ajustar dinamicamente o peso conforme o comportamento relativo das previsões;
                                - Produzir uma previsão combinada que tenda a reduzir o erro médio (por exemplo, MAE ou RMSE) em relação aos valores reais observados.
                                
                                DADOS DE ENTRADA (NOVAS PREDIÇÕES):
                                Após o aprendizado, você receberá novos dados no formato <predictions>, contendo apenas as predições dos mesmos modelos (sem o valor real):

                            Seu trabalho é gerar a previsão combinada mais provável, baseada no padrão aprendido nos exemplos anteriores.

                                PADRÃO DE RESPOSTA:
                                1. Sua resposta deve conter uma lista numérica no atributo result, representando as previsões combinadas simulando o valor real estimado e uma breve descrição do que foi aplicado, como quais modelos foram bons em quais pontos no atributo em description.
                                2. Lembre se que existem apenas os 12 pontos de cada modelo previsto.
                                3. Observará cada ponto de cada modelo nos exemplos e entenderá qual modelo é o melhor para o ponto em questão. Ou seja, apenas um modelo pode ser bom no ponto 0, ponto 1 e assim por diante e dirá em quais exemplos cada modelo foi o melhor pela quantidade de vitorias. Exemplo: Modelo RF venceu no primeiro ponto para o primeiro example, modelo ARIMA venceu no segundo ponto no segundo example e no terceiro...
                                4. A lista deve ter o mesmo tamanho das listas de entrada.
                                5. Cada valor da lista deve ser um float, representando a previsão final ajustada.
                                6. Nenhum texto adicional deve ser incluído — apenas a estrutura formatada.

                                DICAS DE COMBINAÇÃO:
                                - Priorize os modelos que apresentaram menor erro absoluto médio nos exemplos anteriores.
                                - Em situações onde houver divergência entre modelos, aproxime-se dos que historicamente tiveram melhor desempenho.
                                - Ajuste a previsão combinada para refletir a tendência média corrigida dos modelos mais precisos, evitando viés sistemático (por exemplo, subestimação constante).

                                RESUMO DO COMPORTAMENTO ESPERADO:
                                - Aprenda com os exemplos o comportamento relativo de cada modelo.
                                - Gere previsões combinadas mais precisas.
                                - Retorne apenas listas numéricas, sem explicações adicionais.
                                """
            ),
        )

        prompt = f"""
        Use todos os {num_examples} exemplos de treinamento abaixo.
        {training_data_content}
        
        Nova predição de exemplo para gerar o valor ponto a ponto de cada modelo:
        {predictions_string}

        """

        df_ade = pd.read_csv(
            # f"./Statistics_and_Seq2Seq/timeseries/mestrado/resultados/ADE/ANP_MONTHLY.csv",
            f"./timeseries/mestrado/resultados/ADE/ANP_MONTHLY.csv",
            sep=";",
        )

        # df_ade = df_ade[df_ade["dataset_index"] == dataset_index].iloc[0]
        # ade_predictions = extract_values(df_ade["predictions"])
        # print(f"ADE predictions: {ade_predictions}")
        # # await agent.aprint_response(prompt)
        response: RunOutput = agent.run(prompt)
        # print(f"Valores reais esperados: {test_values}")
        # print(f"Media simples dos modelos: {mean_predictions}")

        # import matplotlib.pyplot as plt
        import numpy as np

        agent_response = response.content or ""
        if not isinstance(agent_response, str):
            agent_response = str(agent_response)
        agent_response = re.sub(
            r"<think>.*?</think>", "", agent_response, flags=re.DOTALL
        )
        agent_response = agent_response.strip()
        print(agent_response)
        agent_predictions = extract_values(response.content.result)
        print(f"final predictions: {agent_predictions}")

        preds_real_array = np.array(agent_predictions)
        preds_real_reshaped = preds_real_array.reshape(1, -1)
        test_reshaped = np.array(test_values).reshape(1, -1)
        smape_result = calculate_smape(preds_real_reshaped, test_reshaped)
        rmse_result = calculate_rmse(preds_real_reshaped, test_reshaped)
        msmape_result = calculate_msmape(preds_real_reshaped, test_reshaped)
        # mase_result = calculate_mase(preds_real_reshaped, test_reshaped, training_set, seasonality)
        mae_result = calculate_mae(preds_real_reshaped, test_reshaped)
        mape_result = mape(test_values, preds_real_array)
        pocid_result = pocid(test_values, preds_real_array)

        print(f"SMAPE: {smape_result[0]}")
        print(f"RMSE: {rmse_result[0]}")
        print(f"mSMAPE: {msmape_result[0]}")
        print(f"MAE: {mae_result[0]}")
        print(f"MAPE: {mape_result}")
        print(f"POCID: {pocid_result}")

        data_serie = {
            "dataset_index": f"{dataset_index}",
            "horizon": 12,
            "regressor": "ANP_MONTHLY",
            "mape": mape_result,
            "pocid": pocid_result,
            "smape": smape_result,
            "rmse": rmse_result,
            "msmape": msmape_result,
            "mae": mae_result,
            "test": [test_values],
            "predictions": [agent_predictions],
            "start_test": "2023-12-31",
            "final_test": final_test,
            # 'training_time': times[0],
            # 'prediction_time': times[1],
        }

        if not os.path.exists(path_csv):
            pd.DataFrame(columns=cols_serie).to_csv(path_csv, sep=";", index=False)

        df_new = pd.DataFrame(data_serie)
        df_new.to_csv(path_csv, sep=";", mode="a", header=False, index=False)

    # x = np.arange(len(test_values))
    # plt.figure(figsize=(10, 6))
    # plt.plot(x, test_values, label="Valores Reais", marker="o")
    # plt.plot(x, mean_predictions, label="Média Simples dos Modelos", marker="o")
    # plt.plot(x, agent_predictions, label="Predições do Agente", marker="o")
    # plt.plot(x, ade_predictions, label="Predições ADE", marker="o")
    # plt.xlabel("Índice")
    # plt.ylabel("Valores")
    # plt.title("Comparação de Predições")
    # plt.legend()
    # plt.grid()
    # plt.savefig(f"{dataset_index}_{final_test}_{model_name}.png")
    # plt.show()


if __name__ == "__main__":
    asyncio.run(main())
