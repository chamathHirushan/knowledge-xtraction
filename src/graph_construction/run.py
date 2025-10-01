# from argparse import ArgumentParser
from graph_construction.edc.edc_framework import EDC
import os
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     # OIE module setting
#     parser.add_argument(
#         "--oie_llm", default="mistralai/Mistral-7B-Instruct-v0.2", help="LLM used for open information extraction."
#     )
#     parser.add_argument(
#         "--oie_prompt_template_file_path",
#         default="./prompt_templates/oie_template.txt",
#         help="Promp template used for open information extraction.",
#     )
#     parser.add_argument(
#         "--oie_few_shot_example_file_path",
#         default="./few_shot_examples/example/oie_few_shot_examples.txt",
#         help="Few shot examples used for open information extraction.",
#     )

#     # Schema Definition setting
#     parser.add_argument(
#         "--sd_llm", default="mistralai/Mistral-7B-Instruct-v0.2", help="LLM used for schema definition."
#     )
#     parser.add_argument(
#         "--sd_prompt_template_file_path",
#         default="./prompt_templates/sd_template.txt",
#         help="Prompt template used for schema definition.",
#     )
#     parser.add_argument(
#         "--sd_few_shot_example_file_path",
#         default="./few_shot_examples/example/sd_few_shot_examples.txt",
#         help="Few shot examples used for schema definition.",
#     )

#     # Schema Canonicalization setting
#     parser.add_argument(
#         "--sc_llm",
#         default="mistralai/Mistral-7B-Instruct-v0.2",
#         help="LLM used for schema canonicaliztion verification.",
#     )
#     parser.add_argument(
#         "--sc_embedder", default="intfloat/e5-mistral-7b-instruct", help="Embedder used for schema canonicalization. Has to be a sentence transformer. Please refer to https://sbert.net/"
#     )
#     parser.add_argument(
#         "--sc_prompt_template_file_path",
#         default="./prompt_templates/sc_template.txt",
#         help="Prompt template used for schema canonicalization verification.",
#     )

#     # Refinement setting
#     parser.add_argument("--sr_adapter_path", default=None, help="Path to adapter of schema retriever.")
#     parser.add_argument(
#         "--sr_embedder", default="intfloat/e5-mistral-7b-instruct", help="Embedding model used for schema retriever. Has to be a sentence transformer. Please refer to https://sbert.net/"
#     )
#     parser.add_argument(
#         "--oie_refine_prompt_template_file_path",
#         default="./prompt_templates/oie_r_template.txt",
#         help="Prompt template used for refined open information extraction.",
#     )
#     parser.add_argument(
#         "--oie_refine_few_shot_example_file_path",
#         default="./few_shot_examples/example/oie_few_shot_refine_examples.txt",
#         help="Few shot examples used for refined open information extraction.",
#     )
#     parser.add_argument(
#         "--ee_llm", default="mistralai/Mistral-7B-Instruct-v0.2", help="LLM used for entity extraction."
#     )
#     parser.add_argument(
#         "--ee_prompt_template_file_path",
#         default="./prompt_templates/ee_template.txt",
#         help="Prompt templated used for entity extraction.",
#     )
#     parser.add_argument(
#         "--ee_few_shot_example_file_path",
#         default="./few_shot_examples/example/ee_few_shot_examples.txt",
#         help="Few shot examples used for entity extraction.",
#     )
#     parser.add_argument(
#         "--em_prompt_template_file_path",
#         default="./prompt_templates/em_template.txt",
#         help="Prompt template used for entity merging.",
#     )

#     # Input setting
#     parser.add_argument(
#         "--input_text",
#         default="",
#         help="input texts to extract KG from, each line contains one piece of text.",
#     )
#     parser.add_argument(
#         "--target_schema_path",
#         default="./schemas/example_schema.csv",
#         help="File containing the target schema to align to.",
#     )
#     parser.add_argument("--refinement_iterations", default=0, type=int, help="Number of iteration to run.")
#     parser.add_argument(
#         "--enrich_schema",
#         action="store_true",
#         help="Whether un-canonicalizable relations should be added to the schema.",
#     )

#     # Output setting
#     parser.add_argument("--output_dir", default="./output/tmp", help="Directory to output to.")
#     parser.add_argument("--logging_verbose", action="store_const", dest="loglevel", const=logging.INFO)
#     parser.add_argument("--logging_debug", action="store_const", dest="loglevel", const=logging.DEBUG)

#     args = parser.parse_args()
#     args = vars(args)
#     edc = EDC(**args)

#     input_text_list = open(args["input_text"], "r").readlines()
#     output_kg = edc.extract_kg(
#         input_text_list,
#         args["output_dir"],
#         refinement_iterations=args["refinement_iterations"],
#     )
def run_edc(
    oie_llm=None,
    oie_prompt_template_file_path="./graph_construction/prompt_templates/oie_template.txt",
    oie_few_shot_example_file_path="./graph_construction/few_shot_examples/example/oie_few_shot_examples.txt",
    sd_llm=None,
    sd_prompt_template_file_path="./graph_construction/prompt_templates/sd_template.txt",
    sd_few_shot_example_file_path="./graph_construction/few_shot_examples/example/sd_few_shot_examples.txt",
    sc_llm=None,
    sc_embedder=None,
    sc_prompt_template_file_path="./graph_construction/prompt_templates/sc_template.txt",
    sr_adapter_path=None,
    sr_embedder=None,
    oie_refine_prompt_template_file_path="./graph_construction/prompt_templates/oie_refine_template.txt",
    oie_refine_few_shot_example_file_path="./graph_construction/few_shot_examples/example/oie_refine_few_shot_examples.txt",
    ee_llm=None,
    ee_prompt_template_file_path="./graph_construction/prompt_templates/ee_template.txt",
    ee_few_shot_example_file_path="./graph_construction/few_shot_examples/example/ee_few_shot_examples.txt",
    em_prompt_template_file_path="./graph_construction/prompt_templates/em_template.txt",
    input_text=None,
    target_schema_path=None,
    refinement_iterations=None,
    enrich_schema=None,
    output_dir=None,
    loglevel=None
):
    """
    Run EDC pipeline with only the arguments provided.
    input_text: either a path to a file with text or a list of strings
    """
    args = {}
    
    print("running edc input text:", input_text)
    
    if oie_llm is not None:
        args["oie_llm"] = oie_llm
    if oie_prompt_template_file_path is not None:
        args["oie_prompt_template_file_path"] = oie_prompt_template_file_path
    if oie_few_shot_example_file_path is not None:
        args["oie_few_shot_example_file_path"] = oie_few_shot_example_file_path

    if sd_llm is not None:
        args["sd_llm"] = sd_llm
    if sd_prompt_template_file_path is not None:
        args["sd_prompt_template_file_path"] = sd_prompt_template_file_path
    if sd_few_shot_example_file_path is not None:
        args["sd_few_shot_example_file_path"] = sd_few_shot_example_file_path

    if sc_llm is not None:
        args["sc_llm"] = sc_llm
    if sc_embedder is not None:
        args["sc_embedder"] = sc_embedder
    if sc_prompt_template_file_path is not None:
        args["sc_prompt_template_file_path"] = sc_prompt_template_file_path

    args["sr_adapter_path"] = sr_adapter_path
    args["sr_embedder"] = sr_embedder
    if oie_refine_prompt_template_file_path is not None:
        args["oie_refine_prompt_template_file_path"] = oie_refine_prompt_template_file_path
    if oie_refine_few_shot_example_file_path is not None:
        args["oie_refine_few_shot_example_file_path"] = oie_refine_few_shot_example_file_path

    if ee_llm is not None:
        args["ee_llm"] = ee_llm
    if ee_prompt_template_file_path is not None:
        args["ee_prompt_template_file_path"] = ee_prompt_template_file_path
    if ee_few_shot_example_file_path is not None:
        args["ee_few_shot_example_file_path"] = ee_few_shot_example_file_path
    if em_prompt_template_file_path is not None:
        args["em_prompt_template_file_path"] = em_prompt_template_file_path

    if input_text is not None:
        args["input_text"] = input_text
    args["target_schema_path"] = target_schema_path
    if refinement_iterations is not None:
        args["refinement_iterations"] = refinement_iterations
    if enrich_schema is not None:
        args["enrich_schema"] = enrich_schema
    if output_dir is not None:
        args["output_dir"] = output_dir
    args["loglevel"]  = False
    if loglevel:
        args["loglevel"] = True

    edc = EDC(**args)

    if isinstance(input_text, str):
        input_text_list = [line.strip() for line in input_text.split("\n") if line.strip()]
    else:
        input_text_list = [input_text]

    output_kg = edc.extract_kg(
        input_text_list,
        args.get("output_dir", "./output"),
        refinement_iterations=args.get("refinement_iterations", 0),
    )
    
    return output_kg