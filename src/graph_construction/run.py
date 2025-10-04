from graph_construction.edc.edc_framework import EDC

def run_edc(
    oie_llm=None,
    schema_llm=None,
    sc_embedder=None,
    sr_adapter_path=None,
    sr_embedder=None,
    input_text=None,
    target_schema=None,
    output_dir=None,
    refinement_iterations=0,
    enrich_schema=True,
    loglevel=True
):
    args = {}

    if input_text or schema_llm or sc_embedder is None:
        return
    
    args["oie_llm"] = oie_llm
    args["schema_llm"] = schema_llm
    args["sc_embedder"] = sc_embedder

    args["sr_adapter_path"] = sr_adapter_path or None
    args["sr_embedder"] = sr_embedder or None
    args["target_schema"] = target_schema or None
    args["enrich_schema"] = enrich_schema
    args["loglevel"] = loglevel

    edc = EDC(**args)

    output_kg, definitions = edc.extract_kg(
        [input_text],
        output_dir,
        refinement_iterations=refinement_iterations,
    )
    
    return output_kg, definitions