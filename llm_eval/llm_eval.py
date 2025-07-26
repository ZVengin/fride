"""
Speaker-identification evaluation script.
"""
import argparse
import json
import os
import sys
import re
import logging
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "./../src"))
from utils import GetCharId
from openai import OpenAI
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

with open('llm_key') as f:
    key = f.read().strip()
client = OpenAI(api_key=key)



def generate_paragraph(context, mode):
    """
    Call GPT-4 to identify the most likely speaker of `utterance`
    in the given `text_snippet`.
    """

    system_msg = (
        "You are an expert in literary analysis. "
        "Given the context of a story,"
        "You are required to write the next paragraph using the specified fiction-writing mode. "
        "The fiction-writing mode includes Action, Description, and Dialogue."
        "   Action: describe the actions performed by characters."
        "   Description: describe the appearance of characters or describe the environment."
        "   Dialogue: depict the spoken exchange between characters."
        "The next paragraph should be returned in JSON: "
        '{"next_paragraph": "<Next Paragraph>"}'
    )
    user_msg = f"""### Context:
{context}

### Fiction-writing mode:
{mode}
"""
    #logger.info(system_msg)
    #logger.info(user_msg)
    response = client.chat.completions.create(
        model='gpt-4.1',
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    )

    content = response.choices[0].message.content.strip()
    pattern = r'\{\s*"next_paragraph"\s*:.*?\}'

    match = re.search(pattern, content)
    if match:
        next_para_dict = json.loads(match.group(0))
        logger.info(next_para_dict)
        return next_para_dict.get("next_paragraph")
    else:
        logger.info("⚠️  Failed to parse model response:", content)
        return None



def eval_dataset(dataset_path, result_path):
    """Run inference over the dataset and save per-instance records."""
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            records = json.load(f)
        logger.info(f"Loading previous results from {result_path}")
        done_ids = {r["paragraph_index"] for r in records}
        todo_dataset = [inst for inst in dataset if inst["paragraph_index"] not in done_ids]
    else:
        records = []
        todo_dataset = dataset

    logger.info(
        f"Original dataset: {len(dataset)}, "
        f"untested subset: {len(todo_dataset)}"
    )
    #todo_dataset = todo_dataset[:5]

    for idx, inst in enumerate(todo_dataset, 1):
        gen_para = generate_paragraph(
            context=inst.get("context"),
            mode=inst.get('mode')
        )

        if gen_para:
            records.append(
                {
                    "paragraph_index": inst.get("paragraph_index"),
                    "generated_paragraph": gen_para,
                    "context": inst.get("context"),
                    'mode': inst.get('mode')
                }
            )

        if idx % 100 == 0 or idx == len(todo_dataset):
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            pct = len(records) / len(dataset) * 100
            logger.info(f"Progress: {pct:.1f}%  [{len(records)}/{len(dataset)}]")






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    args = parser.parse_args()

    eval_dataset(args.dataset_path, args.result_path)
