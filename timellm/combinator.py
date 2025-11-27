from agno.agent import Agent
from agno.tools import tool
from agno.models.ollama import Ollama
import os
import sys
import json
import re
import numpy as np
from typing import List
from pydantic import BaseModel, Field
# from combination_functions import full_combination
from sklearn.metrics import mean_absolute_percentage_error as mape

def full_combination(predictions):
    """
    Combines all predictions using simple average.
    predictions: dict of model_name: prediction_array
    Returns combined prediction as numpy array
    """
    if not predictions:
        raise ValueError("No predictions provided")
    all_preds = list(predictions.values())
    return np.mean(all_preds, axis=0)

def clean_json_string(text: str) -> dict:
    text = text.strip()
    
    text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'^```\s*', '', text)
    
    text = re.sub(r'\s*```$', '', text)
    
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Erro ao parsear JSON: {e}")
        print(f"Texto após limpeza: {text[:200]}...")
        raise

class CombinationResult(BaseModel):
    description: str = Field(
        ..., 
        description="Descrição do método de combinação utilizado"
    )
    result: List[float] = Field(
        ..., 
        description="Lista com as predições combinadas resultantes"
    )
    selected_models: str = Field(
        ..., 
        description="Modelos selecionados para combinação"
    )

@tool
def calculate_metrics_tool(validation_test: list, validation_predictions: dict) -> dict:
    """
        function to calculate metrics for each model given the validation test and validation predictions
        MAPE: the lowest is best
        SMAPE: the lowest is best
        RMSE: the lowest is best
        POCID: the highest is best
    """
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import all_functions
    
    print(f"\n[TOOL CALL] calculate_metrics_tool called")
    
    results = {}
    y_true = np.array(validation_test)
    
    for model_name, y_pred_list in validation_predictions.items():
        y_pred = np.array(y_pred_list)
        
        # Ensure lengths match
        min_len = min(len(y_true), len(y_pred))
        if min_len == 0:
            continue
            
        curr_y_true = y_true[:min_len]
        curr_y_pred = y_pred[:min_len]
        
        # MAPE
        mape_value = mape(curr_y_true, curr_y_pred)
        
        rmse = all_functions.calculate_rmse(curr_y_pred.reshape(1, -1), curr_y_true.reshape(1, -1))[0]
        
        # SMAPE
        smape = all_functions.calculate_smape(curr_y_pred.reshape(1, -1), curr_y_true.reshape(1, -1))[0]

        # POCID
        pocid = all_functions.pocid(curr_y_true, curr_y_pred)

        results[model_name] = {
            "MAPE": float(mape_value),
            "RMSE": float(rmse),
            "SMAPE": float(smape),
            "POCID": float(pocid)
        }
        
    print(f"[TOOL RESULT] Calculated metrics for {len(results)} models")
    print(results)
    return json.dumps(results, indent=2)


@tool
def selective_combine_tool(predictions: dict, models_to_combine: list) -> dict:
    """
    Combines predictions from specified models using mean averaging.
    
    Args:
        predictions: The final predictions from each differnte models
        models_to_combine: list of model names (strings) to combine from predictions dict
    
    Returns:
        dict with 'result' (list), 'models_used' (list), and 'method' (str)
    """
    print(f"\n[TOOL CALL] selective_combine_tool called")
    print(f"[TOOL INFO] Models to combine: {models_to_combine}")
    
    valid_models = [m for m in models_to_combine if m in predictions]
    
    if not valid_models:
        print(f"[TOOL WARNING] No valid models found. Using all available models.")
        valid_models = list(predictions.keys())
    
    if len(valid_models) < len(models_to_combine):
        missing = set(models_to_combine) - set(valid_models)
        print(f"[TOOL WARNING] Models not found in predictions: {missing}")
    
    print(f"[TOOL INFO] Combining {len(valid_models)} models: {valid_models}")
    
    pred_arrays = {k: np.array(v) for k, v in predictions.items() if k in valid_models}
    
    combined = full_combination(pred_arrays)
    print(f"[TOOL RESULT] Combined {len(combined)} predictions (Mean of {len(valid_models)} models)")
    
    return {
        "result": combined.tolist(),
        "models_used": valid_models,
        "method": f"Mean combination of {len(valid_models)} selected models"
    }



def agent_combinator(model_id: str, temperature: float):
    instructions = """You are an expert time series analysis assistant with access to tools.
            When given validation series and model predictions from this validation, you must:
            1. First, use calculate_metrics_tool to evaluate all models using the validation data
            2. Then, choose the best models to use in selective_combine_tool based on the metrics that you think performs best to combine the best performing models using the actual final predictions, pass it as models_to_combine parameter.
            3. After the result from calculate_metrics_tool, Provide a final output in text format in json structure with description and the final combined predictions as result parameter
            Example: 
                description: The models selected were **svr**, **ARIMA**, and **THETA** based on the **SMAPE** metric, as it provided the most consistent performance across the test data. These models were chosen because they exhibited the lowest SMAPE values compared to other models. The predictions from these models have been combined using the **selective_combine_tool** with a **mean-based approach** to produce the final refined predictions. 
                
                result: [16331.74, 12108.99, 17162.63, 17950.0, 17194.63, 17772.72, 18213.79, 18500.46, 18834.40, 18404.49, 18763.25, 18496.37]

            IMPORTANT: Actually CALL the tools using function calling. Do not just describe what you would do."""
    
    return Agent(
        model=Ollama(id=model_id, options={"temperature": temperature, "seed": 42, "num_ctx": 8192}),
        tools=[calculate_metrics_tool, selective_combine_tool],
        instructions=instructions,
        markdown=True,
    )
    

def simple_agent(validation_test, validation_predictions, final_test_predictions):
    agent = agent_combinator(
        model_id="qwen3:14b",
        temperature=0.1
    )
    
    print("=" * 80)
    print("AGENT EXECUTION")
    print("=" * 80)
    print(f"Model: {agent.model.id}")
    print(f"Models: {len(final_test_predictions)}")
    print("=" * 80 + "\n")
    
    prompt = f"""You have a set of tools to call and use. Your task is to analyse the results from each tool, provide an understand and call other tool that you think is most useful.
    Your data is the validation test data, which are the actual values from the validation, and the validation predictions from different models.

    You also have the final test predictions from different models which will use to combine after selected the best models to combine generate from validation data.
    Validation data (actual values): {validation_test}

    Validation predictions from different models:
    {json.dumps(validation_predictions, indent=2)}


    Final test predictions from different models that you will be passed to selective_combine_tool:
    {json.dumps(final_test_predictions, indent=2)}

    Your task:
    1. First, you need to calculate the performance metrics for each model using the function call calculate_metrics_tool with the validation test and validation predictions.
    2. Select the models with the best perfomance based on the metrics you choose and pass it to selective_combine_tool as models_to_combine parameter and final_test_predictions as predictions parameter.
    3. The selective_combine_tool will combine the predictions from the selected models and the predictions from the final test data.
    4. Only After the result from selective_combine_tool tool call, Provide a final output in text format in json structure with summary as description saying which models were selected and why and the combined predictions as result
    Example: 
        description: The models selected were **svr**, **ARIMA**, and **THETA** based on the **SMAPE** metric, as it provided the most consistent performance across the test data. These models were chosen because they exhibited the lowest SMAPE values compared to other models. The predictions from these models have been combined using the **selective_combine_tool** with a **mean-based approach** to produce the final refined predictions. 
        
        result: [16331.74, 12108.99, 17162.63, 17950.0, 17194.63, 17772.72, 18213.79, 18500.46, 18834.40, 18404.49, 18763.25, 18496.37]


    Please use your tools to complete this task."""
        
    
    print("Sending prompt to agent...")
    print("-" * 80)
    try:
        response = agent.run(prompt)
        # print("\nAGENT RESPONSE:\n")
        # print("---- PURE RESPONSE ---")
        print(response.content)
        output = clean_json_string(response.content)
        # print("----- CLEAN RESPONSE -----")
        # print("Description: ", output.get("description", "N/A"))
        # print("Combination: ", output.get("result", []))
        return output.get("description", ""), output.get("result", [])
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    pass

if __name__ == "__main__":
    validation_test = [17656.623, 19507.078, 15680.762, 19775.546, 13736.136, 17221.028, 18352.012, 20327.377, 21175.424, 18516.637, 19864.665, 18523.176]

    validation_predictions = {
            "ARIMA": [20189.34563269, 20192.85789627, 20092.93080571, 20125.30269528, 19897.04910318, 20355.62201653, 21603.84938594, 20580.46644655, 21348.37952553, 20451.24241379, 20830.78699143, 20366.88668088],
            "ETS": [18062.62317529, 17393.33066554, 15407.84271242, 17092.38074023, 16162.45141857, 16844.88650947, 17782.62518231, 18540.99028846, 19297.52887647, 18610.390685, 19436.7535238, 19007.89060085],
            "THETA": [17866.08934355, 17144.80877427, 15254.82768456, 17103.34824538, 16019.72145045, 16809.28489874, 17700.12233432, 18537.68322061, 19161.68786003, 18545.03624378, 19271.48287791, 18913.88997639],
            "svr": [18759.42203864, 18759.42210619, 18759.4221062, 18759.4221062, 18759.42210639, 18759.42210538, 18759.42213586, 18759.4221062, 18759.4221062, 18759.4221062, 18759.4221062, 18759.4221062],
            "rf": [17103.19423397, 17803.67250428, 17731.06974653, 18104.01435002, 18000.64727707, 18303.83074421, 18103.89545828, 18113.67717747, 18026.40149462, 18120.85642935, 17798.38598629, 17688.20732412],
            "catboost": [16997.71257262, 18223.88166564, 17910.64906155, 18052.06112336, 17395.51510697, 17464.68223325, 16751.90487253, 15703.21729542, 15213.6946995, 15455.24530211, 16404.94868817, 18586.2942714],
            "CWT_svr": [18759.51542668, 18515.20822027, 18295.9953686, 18150.54620463, 18073.18622798, 18188.19369318, 18151.85355396, 18048.70977966, 17975.13513241, 18000.12467132, 17998.57014806, 17976.45072006],
            "DWT_svr": [18759.51542668, 18515.20822027, 18295.9953686, 18150.54620463, 18073.18622798, 18188.19369318, 18151.85355396, 18048.70977966, 17975.13513241, 18000.12467132, 17998.57014806, 17976.45072006],
            "FT_svr": [16986.4041512, 18005.59142402, 18099.29958901, 18571.83507727, 18548.55281724, 18406.59623913, 18653.85029216, 18370.60372238, 17420.79279064, 17460.84211587, 17347.69927974, 17141.04602255],
            "CWT_rf": [16810.50631223, 16970.34844394, 16968.19324107, 16680.46123042, 16999.2317529, 16553.27749375, 16781.29620828, 16431.03309906, 16644.58693919, 16396.60590851, 16386.40946308, 16481.18908182],
            "DWT_rf": [16726.88958047, 17370.9128624, 16684.84138993, 17269.19743572, 17246.71066812, 17060.49243543, 17086.71343359, 16896.54214381, 16474.99586124, 16581.70867416, 16356.79727089, 16483.25983542],
            "FT_rf": [15737.96054126, 16124.04505268, 17410.61249198, 17342.36712357, 17898.41309604, 18194.83951578, 18306.64199349, 18295.73810922, 17997.28435946, 17762.59600298, 16908.27926607, 16875.18971294],
            "CWT_catboost": [18021.92606255, 17817.83554492, 18011.85713323, 17890.44975533, 17949.9677973, 18120.2264075, 18115.2101686, 18060.46296962, 17863.39983706, 17781.64505428, 17662.88374363, 17749.57771547],
            "DWT_catboost": [18114.39598359, 17960.48813212, 17783.57486875, 17688.82118656, 17585.03278393, 17745.14227767, 17638.62737912, 17646.17618313, 17643.0873276, 17652.8525517, 17609.07633908, 17616.40332561],
            "FT_catboost": [16413.09711353, 17307.99150984, 16641.3797852, 17893.33707602, 16333.22457947, 17197.48149865, 17136.21290436, 17786.45849121, 17253.56123353, 17112.93159883, 16913.48233282, 16626.96867976]
            }
        
    test_series = [22026.58, 21691.041, 18155.248, 17892.518, 18338.645, 11259.7, 18852.424, 17437.912, 17794.103, 13666.352, 14124.142, 12543.652]
    predictions = {
            "ARIMA": [19197.09616343, 19218.19911323, 18289.75028673, 18991.53921425, 17830.29109381, 18676.75281918, 18991.222043, 19210.49939444, 19533.24147234, 18841.98662861, 19274.82557375, 18851.34984505],
            "ETS": [18251.1580763, 17924.35960825, 16248.24412114, 17766.15931347, 16819.15680994, 17510.64335004, 18319.57207669, 18979.65520757, 19563.43835807, 18870.87997772, 19539.15669582, 19149.92014317],
            "THETA": [18345.6940208, 17874.10983595, 16222.97612274, 17834.21723458, 16755.47385476, 17504.31572072, 18362.58535134, 18904.91805159, 19501.2955687, 18887.74346103, 19515.8698591, 19162.3682469],
            "svr": [17452.4221411, 17234.64753482, 16975.16681594, 17024.21102249, 16998.12236703, 17137.08981868, 17287.56864945, 17385.94955304, 17468.65515164, 17483.75247101, 17499.0596454, 17475.40451716],
            "rf": [17594.56713935, 16988.53851451, 15220.13799115, 16405.03819754, 16229.28107273, 16487.60403589, 17313.70069943, 17700.77143317, 18067.47250334, 18644.77847774, 18686.94990243, 18284.22764055],
            "catboost": [17480.08436303, 17300.88465267, 14527.8578392, 16118.70803821, 16789.43842464, 16225.13660246, 18107.8765838, 18216.16281705, 18866.88706808, 19729.19736026, 20467.58901349, 19236.18076527],
            "CWT_svr": [17473.40367069, 17027.65583789, 16479.605163, 16581.08392297, 16392.44084997, 16969.30862415, 17189.00332273, 17331.74512162, 17340.43932507, 17247.77175877, 17135.69334263, 16981.97050238],
            "DWT_svr": [17926.94347244, 17258.82698246, 15882.86857631, 17277.4223869, 16914.33628065, 16568.21928437, 17860.31031934, 17692.72656005, 18223.70572647, 18016.41987286, 17756.22133002, 17672.8968208],
            "FT_svr": [17824.29082877, 17374.40021011, 15444.34116769, 16953.06272067, 17102.35871569, 16499.87199132, 17537.28623046, 17877.09881465, 17625.97791532, 18124.59991469, 18276.50134908, 17877.46019979],
            "CWT_rf": [16567.05519459, 16059.52855682, 15258.51218364, 16467.14154321, 15747.6389172, 16229.90071796, 16424.2071959, 16586.9576747, 16970.48077076, 16522.73725021, 16235.52296597, 16107.1922324],
            "DWT_rf": [18098.46198988, 16985.5179056, 15451.3318093, 15725.41631389, 14862.19180381, 15917.73977454, 16228.85838407, 16255.4209196, 16832.75047669, 16778.2652245, 16604.76938011, 16289.94931507],
            "FT_rf": [17348.61122703, 16849.82524113, 15096.18413635, 15324.5564488, 15305.54604854, 15838.24011944, 16340.51110191, 16686.36627673, 16886.05694192, 16705.58555394, 16368.801103, 16160.07672048],
            "CWT_catboost": [17377.02753016, 16427.94456663, 14510.93589461, 15876.21508147, 15308.66034944, 15532.97247311, 16428.67357426, 16937.23082187, 17263.57037257, 16645.22903114, 16518.70702744, 16039.16339444],
            "DWT_catboost": [17842.93000058, 17500.06127439, 16702.69910431, 16955.12667547, 16820.62850256, 17339.22385666, 17528.69917841, 17642.78761942, 17726.55757098, 17571.69823637, 17548.1240138, 17409.61835574],
            "FT_catboost": [17729.70542317, 17367.36165659, 16797.76082268, 17016.07584365, 16779.51393948, 17312.56091945, 17543.29106472, 17704.00934383, 17732.31867146, 17519.12232138, 17485.72105834, 17363.68890376]
}
    
    agent = agent_combinator(
        model_id="qwen3:14b",
        temperature=0.0
    )
    
    print("=" * 80)
    print("AGENT EXECUTION")
    print("=" * 80)
    print(f"Model: {agent.model.id}")
    print(f"Test points: {len(test_series)}")
    print(f"Models: {len(predictions)}")
    print("=" * 80 + "\n")
    
    prompt = f"""You have a set of tools to call and use. Your task is to analyse the results from each tool, provide an understand and call other tool that you think is most useful.
Your data is the validation test data, which are the actual values from the validation, and the validation predictions from different models.

You also have the final test predictions from different models which will use to combine after selected the best models to combine generate from validation data.
Validation data (actual values): {validation_test}

Validation predictions from different models:
{json.dumps(validation_predictions, indent=2)}


Final test predictions from different models that you will be passed to selective_combine_tool:
{json.dumps(predictions, indent=2)}

Your task:
1. First, you need to calculate the performance metrics for each model using calculate_metrics_tool with the validation test and validation predictions.
2. Select the models with the best perfomance based on the metrics you choose and pass it to selective_combine_tool as models_to_combine parameter and final_test_predictions as predictions parameter.
3. The selective_combine_tool will combine the predictions from the selected models and the predictions from the final test data.
4. Provide a final output in text format in json structure with summary as description saying which models were selected and why and the combined predictions as result
Example: 
    description: The models selected were **svr**, **ARIMA**, and **THETA** based on the **SMAPE** metric, as it provided the most consistent performance across the test data. These models were chosen because they exhibited the lowest SMAPE values compared to other models. The predictions from these models have been combined using the **selective_combine_tool** with a **mean-based approach** to produce the final refined predictions. 
    
    result: [16331.74, 12108.99, 17162.63, 17950.0, 17194.63, 17772.72, 18213.79, 18500.46, 18834.40, 18404.49, 18763.25, 18496.37]


Please use your tools to complete this task."""
    
    print("Sending prompt to agent...")
    print("-" * 80)
    try:
        response = agent.run(prompt)
        print("\nAGENT RESPONSE:\n")
        print("---- PURE RESPONSE ---")
        print(response.content)
        output = clean_json_string(response.content)
        print("----- CLEAN RESPONSE -----")
        print("Description: ", output.get("description", "N/A"))
        print("Combination: ", output.get("result", []))
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()