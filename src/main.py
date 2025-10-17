import os
import pandas as pd
from graph_construction.run import run_edc 

from extraction.candidate_llms import is_model_gemini, ask_gemini_model
from extraction.QA_datasets import load_mesaqa
df = load_mesaqa()

candidate_llms = ["gemini-2.5-flash"] #TODO
oit_llm = "gemini-2.5-flash"        #TODO
schema_llm = "gemini-2.5-flash"     #TODO
sc_embedder="all-mpnet-base-v2" #TODO #intfloat/e5-mistral-7b-instruct

edc_path = os.path.join(os.getcwd(), "./graph_construction")

def build_graphs(gold, context, llm_answer, oie_llm=oit_llm, schema_llm=schema_llm):
    gold_kg, gold_def = run_edc(
        oie_llm=oie_llm,
        schema_llm=schema_llm,
        sc_embedder=sc_embedder,
        input_text=[gold]
    )
    llm_kg, llm_def = run_edc(
        oie_llm=oie_llm,
        schema_llm=schema_llm,
        sc_embedder=sc_embedder,
        input_text=[llm_answer],
        target_schema=gold_def
    )
    context_kg, context_def = run_edc(
        oie_llm=oie_llm,
        schema_llm=schema_llm,
        sc_embedder=sc_embedder,
        input_text=[context],
        target_schema=llm_def
    )
    return gold_kg, llm_kg, context_kg

################################

for llm in candidate_llms:
    print(f"Processing all rows with LLM {llm}...")

    # Ensure output folder exists
    output_path = os.path.join(edc_path, "output")
    os.makedirs(output_path, exist_ok=True)

    # CSV file path for this LLM
    csv_file = os.path.join(output_path, f"{llm.replace('/', '_')}_kg_results.csv")

    for idx, row in df.iterrows():
        if idx >= 3:# TODO
            break
        llm_answer = ""

        if is_model_gemini(llm):
            llm_answer = ask_gemini_model(row["question"] + " " + row["context"], model=llm)
            print("LLM answer:", llm_answer)
        else:
            print(f"LLM {llm} not supported yet, skipping...")
            continue

        # Run pipeline
        kg_gold, kg_llm, kg_context = build_graphs(row["answer"], row["context"], llm_answer=llm_answer)

        # Create a DataFrame for this row
        df_row = pd.DataFrame([{
            "row_id": idx,
            "question": row["question"],
            "gold_answer": row["answer"],
            "kg_gold": str(kg_gold),
            "llm_answer": llm_answer,
            "kg_llm": str(kg_llm),
            "context": row["context"],
            "kg_context": str(kg_context)
        }], columns=[
            "row_id", "question", "gold_answer", "kg_gold", 
            "llm_answer", "kg_llm", "context", "kg_context"
        ])

        # Append to CSV dynamically
        if os.path.exists(csv_file):
            df_row.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df_row.to_csv(csv_file, index=False)

        print(f"Appended results for row {idx} to {csv_file}")
