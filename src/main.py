import shutil
import subprocess
import os
import pandas as pd
from graph_construction.run import run_edc 

from extraction.candidate_llms import is_model_gemini, ask_gemini_model
from extraction.pubmed import load_pubmedqa
df = load_pubmedqa()

candidate_llms = ["gemini-2.5-flash"] #TODO
use_llm = "gemini-2.5-flash"        #TODO

edc_path = os.path.join(os.getcwd(), "./graph_construction")

def run_edc_pipeline(row, llm_answer, llm="gemini-2.5-flash"):
    """
    Run EDC on long_answer, LLM answer, and context_clean of a row.
    `row` is expected to have attributes: long_answer, context_clean
    `llm` is the LLM to use for all steps.
    """

    # Step 1: long_answer
    output_1 = run_edc(
        oie_llm=llm,
        sd_llm=llm,
        sc_llm=llm,
        sc_embedder="intfloat/e5-mistral-7b-instruct",
        ee_llm=llm,
        input_text=[row.long_answer],  # wrap in list
        enrich_schema=True,
        output_dir="graph_construction/output",
    )

    # Step 2: llm_answer
    output_2 = run_edc(
        oie_llm=llm,
        sd_llm=llm,
        sc_llm=llm,
        sc_embedder="intfloat/e5-mistral-7b-instruct",
        ee_llm=llm,
        input_text=[llm_answer] if hasattr(row, "llm_answer") else [],
        enrich_schema=True,
        target_schema_path="graph_construction/output/schema_definitions.csv",
        output_dir="graph_construction/output",
    )

    # Step 3: context_clean
    output_3 = run_edc(
        oie_llm=llm,
        sd_llm=llm,
        sc_llm=llm,
        sc_embedder="intfloat/e5-mistral-7b-instruct",
        ee_llm=llm,
        input_text=[row.context_clean] if hasattr(row, "context_clean") else [],
        enrich_schema=True,
        target_schema_path="graph_construction/output/schema_definitions.csv",
        output_dir="graph_construction/output",
    )

    return output_1, output_2, output_3
################################

for llm in candidate_llms:
    print(f"Processing all rows with LLM {llm}...")

    # Ensure output folder exists
    output_path = os.path.join(edc_path, "output")
    os.makedirs(output_path, exist_ok=True)

    # CSV file path for this LLM
    csv_file = os.path.join(output_path, f"{llm.replace('/', '_')}_kg_results.csv")

    for row in df.itertuples():
        llm_answer = ""
        if is_model_gemini(llm):
            print("Model is gemini, asking gemini model for answer...")
            llm_answer = ask_gemini_model(row.question + " " + row.context_clean, model=llm)
            print("LLM answer:", llm_answer)
        else:
            print(f"LLM {llm} not supported yet.")
            break

        # Run pipeline
        kg_gold, kg_llm, kg_context = run_edc_pipeline(row, llm_answer=llm_answer, llm=llm)

        # Create a DataFrame for this row
        df_row = pd.DataFrame([{
            "kg_gold": str(kg_gold),
            "kg_llm": str(kg_llm),
            "kg_context": str(kg_context)
        }])

        # Append to CSV dynamically
        if os.path.exists(csv_file):
            df_row.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df_row.to_csv(csv_file, index=False)

        print(f"Appended results for row {row.Index} to {csv_file}")
