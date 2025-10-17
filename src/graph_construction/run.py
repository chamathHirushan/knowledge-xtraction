from graph_construction.edc.edc_framework import EDC

def load_edc(
    oie_llm=None,
    schema_llm=None,
    sc_embedder=None,
    sr_adapter_path=None,
    sr_embedder=None,
    enrich_schema=True,
    loglevel=True
):
    args = {}

    if schema_llm is None or sc_embedder is None:
        print("schema_llm, and sc_embedder are required parameters.", schema_llm, sc_embedder)
        return
    
    args["oie_llm"] = oie_llm
    args["schema_llm"] = schema_llm
    args["sc_embedder"] = sc_embedder

    args["sr_adapter_path"] = sr_adapter_path or None
    args["sr_embedder"] = sr_embedder or None
    args["target_schema"] = None
    args["enrich_schema"] = enrich_schema
    args["loglevel"] = loglevel

    edc = EDC(**args)
    return edc

def run_edc(
    edc,
    input_text,
    output_dir=None,
    refinement_iterations=0,
):
    if edc is None or input_text is None or output_dir is None:
        print("edc, input_text, and output_dir are required parameters.", edc, input_text, output_dir)
        return
    
    output_kg, definitions = edc.extract_kg(
        [input_text],
        output_dir,
        refinement_iterations=refinement_iterations,
    )
    
    return output_kg, definitions