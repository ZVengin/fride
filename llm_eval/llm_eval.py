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



def identify_speaker(text_snippet, utterance, characters):
    """
    Call GPT-4 to identify the most likely speaker of `utterance`
    in the given `text_snippet`.
    """
    character_list = "\n".join(
        f"{name}: {', '.join(aliases) if isinstance(aliases, list) else aliases}"
        for name, aliases in characters.items()
    )

    system_msg = (
        "You are an expert in literary analysis. "
        "Given context, utterances, and a dictionary containing character names and their corresponding aliases, "
        "return the most likely speaker of the utterances in JSON: "
        '{"speaker": "<Character Name>"}'
    )
    user_msg = f"""### Context:
{text_snippet}

### Utterances:
{utterance}

### Character name and their corresponding aliases:
{character_list}
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
    pattern = r'\{\s*"speaker"\s*:\s*"([^"]+)"\s*\}'

    match = re.search(pattern, content)
    if match:
        speaker_dict = {"speaker": match.group(1)}
        logger.info(speaker_dict)
        return speaker_dict.get("speaker")
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
        done_ids = {r["id"] for r in records}
        todo_dataset = [inst for inst in dataset if inst["id"] not in done_ids]
    else:
        records = []
        todo_dataset = dataset

    logger.info(
        f"Original dataset: {len(dataset)}, "
        f"untested subset: {len(todo_dataset)}"
    )
    #todo_dataset = todo_dataset[:5]

    for idx, inst in enumerate(todo_dataset, 1):
        # Build full context text
        para_texts = [
            p.get("paragraph")
            for p in (
                inst.get("preceding_paragraphs", [])
                + inst.get("dialogue", [])
                + inst.get("succeeding_paragraphs", [])
            )
        ]
        text = "\n".join(t for t in para_texts if t)  # drop Nones

        # The target utterance we need to attribute
        utterance = [qd['quote'] for qd in inst["dialogue"][0]["utterance"]]
        utterance = "\n".join(utterance)
        qids = [qd['quote_id'] for qd in inst["dialogue"][0]["utterance"]]
        labels = [inst["dialogue"][0]["utterance"][0]['speaker']]
        character_dict = {aliases[0]:aliases for aliases in inst['character']['id2names'].values()}

        speaker = identify_speaker(
            text_snippet=text,
            utterance=utterance,
            characters=character_dict,
        )

        if speaker:
            records.append(
                {
                    "id": inst["id"],
                    "predict_label": [speaker],
                    "predict_label_id": [GetCharId(inst["character"], speaker)],
                    "label": labels,
                    "label_id": [GetCharId(inst["character"], label) for label in labels],
                    "quote_id": [qids]
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
