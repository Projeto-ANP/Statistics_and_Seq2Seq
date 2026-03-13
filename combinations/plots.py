from . import aux
def plot_combination_results(dataset_name, combinations_to_plot, output_dir="plots"):
    """Plot all specified combinations on a single figure per dataset_index.

    For each unique dataset_index contained in any of the combination CSVs, the
    function will build a single plot showing the test series plus the
    predictions produced by each combination.  Only one image is saved per
    dataset_index (not per combination).
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    os.makedirs(output_dir, exist_ok=True)

    # collect all dataframes indexed by combination
    comb_dfs = {}
    all_indices = set()
    for combination in combinations_to_plot:
        path = f"./timeseries/mestrado/resultados/{combination}/{dataset_name}.csv"
        if not os.path.exists(path):
            print(f"warning: missing file for {combination}")
            continue
        df = pd.read_csv(path, sep=";")
        comb_dfs[combination] = df
        all_indices.update(df["dataset_index"].unique())

    # for each dataset_index create a single plot
    for idx in sorted(all_indices):
        plt.figure(figsize=(12, 6))
        test_series = None
        metric_labels = []
        metric_values = []
        for combination, df in comb_dfs.items():
            row = df[df["dataset_index"] == idx]
            if row.empty:
                continue
            row = row.iloc[0]
            preds = aux.extract_values(row.get("predictions", ""))
            # format metrics
            smape_result = row.get("smape", None)
            rmse_result = row.get("rmse", None)
            pocid_result = row.get("pocid", None)
            if smape_result is not None:
                smape_result = f"{smape_result:.4f}"
            else:
                smape_result = "N/A"
            if rmse_result is not None:
                rmse_result = f"{rmse_result:.4f}"
            else:
                rmse_result = "N/A"
            if pocid_result is not None:
                pocid_result = f"{pocid_result:.2f}"
            else:
                pocid_result = "N/A"

            metric_labels.append(combination)
            metric_values.append((smape_result, rmse_result, pocid_result))

            if test_series is None:
                test_series = aux.extract_values(row.get("test", ""))
                plt.plot(test_series, label="test", color="black")
            plt.plot(preds, linestyle="--", label=f"{combination}")

        # first legend for lines
        line_legend = plt.legend(loc='upper left')
        plt.gca().add_artist(line_legend)
        # build metric legend entries
        from matplotlib.lines import Line2D
        metric_handles = [Line2D([0],[0], linestyle='') for _ in metric_labels]
        metric_texts = [f"{lbl}: smape={vals[0]}, rmse={vals[1]}, pocid={vals[2]}" for lbl, vals in zip(metric_labels, metric_values)]
        plt.legend(metric_handles, metric_texts, loc='upper right', title='metrics')

        plt.title(f"Combined predictions for {dataset_name}, idx={idx}")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.grid()

        filename = f"./{dataset_name}/combined_{idx}.png"
        os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    print(f"plots saved in {output_dir}")


if __name__ == "__main__":
    combinations_to_plot = ["mean", "dba", "median", "orchestrator_llm_v1_qwen3.5"]
    dataset_name = "ETTH1"
    plot_combination_results(dataset_name, combinations_to_plot)
    