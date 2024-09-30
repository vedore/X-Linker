"""
    Calculate the Top-k accuracy for each value of k in topk_values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the columns 'code' and 'codes'.
    topk_values : list of int
        List of k values for which to calculate the Top-k accuracy.

    Returns
    -------
    dict
        A dictionary with k values as keys and their corresponding
        accuracies as values.
"""

def calculate_topk_accuracy(df, topk_values):
    # Inicializar diccionario para almacenar los resultados
    topk_accuracies = {k: 0 for k in topk_values}

    for index, row in df.iterrows():
        true_code = row["code"]
        predicted_codes = row["codes"]

        if type(predicted_codes) == str:
            to_add = predicted_codes.strip("[").strip("]").strip("'")
            predicted_codes = [to_add]

        seen = set()
        unique_candidates = [
            x for x in predicted_codes if not (x in seen or seen.add(x))
        ]

        for k in topk_values:
            if true_code in unique_candidates[:k]:
                topk_accuracies[k] += 1

    total_rows = len(df)
    for k in topk_values:
        topk_accuracies[k] = topk_accuracies[k] / total_rows

    return topk_accuracies