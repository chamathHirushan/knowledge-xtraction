import os
import pandas as pd
from graph_construction.run import run_edc 
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
from extraction.candidate_llms import is_model_gemini, ask_gemini_model
from extraction.QA_datasets import load_mesaqa
df = load_mesaqa()

candidate_llms = ["gemini-1.5-flash"] #TODO
oit_llm = "mistralai/Mistral-7B-Instruct-v0.2"#"gemini-2.5-flash"        #TODO
schema_llm = "mistralai/Mistral-7B-Instruct-v0.2"#"gemini-2.5-flash"     #TODO

edc_path = os.path.join(os.getcwd(), "./graph_construction")

def build_graphs(gold, context, llm_answer, oie_llm="gemini-2.5-flash", schema_llm="gemini-2.5-flash"):
    gold_kg, gold_def = run_edc(
        oie_llm=oie_llm,
        schema_llm=schema_llm,
        sc_embedder="intfloat/e5-mistral-7b-instruct",
        input_text=[gold]
    )
    llm_kg, llm_def = run_edc(
        oie_llm=oie_llm,
        schema_llm=schema_llm,
        sc_embedder="intfloat/e5-mistral-7b-instruct",
        input_text=[llm_answer],
        target_schema=gold_def
    )
    context_kg, context_def = run_edc(
        oie_llm=oie_llm,
        schema_llm=schema_llm,
        sc_embedder="intfloat/e5-mistral-7b-instruct",
        input_text=[context],
        target_schema=llm_def
    )
    return gold_kg, llm_kg, context_kg

################################
columns = [
    "row_id",
    "question",
    "context",
    "gold_answer",
    "llm_name",
    "llm_answer",
    "kg_gold",
    "kg_llm",
    "kg_context"
]

for llm in candidate_llms:
    print(f"Processing all rows with LLM: {llm}")

    # Ensure output folder exists
    output_path = os.path.join(edc_path, "output")
    os.makedirs(output_path, exist_ok=True)

    # Define CSV output path for each LLM
    csv_file = os.path.join(output_path, f"{llm.replace('/', '_')}_kg_results.csv")

    # Create file with headers if it doesn't exist
    if not os.path.exists(csv_file):
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

    for idx, row in df.iterrows():
        # Get LLM answer
        if is_model_gemini(llm):
            prompt = f"{row['question']} {row['context']}"
            llm_answer = ask_gemini_model(prompt, model=llm)
            print("LLM answer:", llm_answer)
        else:
            print(f"LLM {llm} not supported yet, skipping...")
            continue
        
        # # Load a small model (example: Llama-3 or Mistral)
        # model_name = "intfloat/e5-mistral-7b-instruct"
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModelForCausalLM.from_pretrained(model_name)

        # question = row["question"] + " " + row["context"]
        # inputs = tokenizer(question, return_tensors="pt")
        
        # outputs = model.generate(
        #     **inputs,
        #     max_new_tokens=256,  # or longer depending on expected answer
        #     do_sample=False
        # )
        # llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print("Answer:", llm_answer)
        # Build knowledge graphs
        kg_gold, kg_llm, kg_context = build_graphs(
            row["answer"], 
            row["context"], 
            llm_answer=llm_answer
        )

        # Construct row data with all column names
        df_row = pd.DataFrame([{
            "row_id": idx,
            "question": row["question"],
            "context": row["context"],
            "gold_answer": row["answer"],
            "llm_name": llm,
            "llm_answer": llm_answer,
            "kg_gold": str(kg_gold),
            "kg_llm": str(kg_llm),
            "kg_context": str(kg_context)
        }], columns=columns)

        # Append to CSV
        df_row.to_csv(csv_file, mode='a', header=False, index=False)

        print(f"✅Appended results for row {idx} to {csv_file}")