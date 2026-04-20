import json
import jsonlines

def convert_chat_to_completion(input_json, output_jsonl):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    with jsonlines.open(output_jsonl, "w") as writer:
        for item in data:
            convs = item["conversations"]
            system = item.get("system", "")

            prompt = ""
            answer = ""

            # 最后一条 gpt 是 completion
            for turn in convs:
                if turn["from"] == "human":
                    prompt += f"{turn['value']}\n"
                elif turn["from"] == "gpt":
                    answer = turn["value"]

            # 拼成 Axolotl 需要的唯一字段 text
            full_text = f"{system}\n{prompt}{answer}"

            writer.write({"text": full_text})


print("Converting subjective...")
convert_chat_to_completion(
    "data/clause_subjective_chat.json",
    "data/clause_subjective_completion.jsonl",
)

print("Converting objective...")
convert_chat_to_completion(
    "data/clause_objective_chat.json",
    "data/clause_objective_completion.jsonl",
)

print("Converting db...")
convert_chat_to_completion(
    "data/db_train.json",
    "data/db_train_completion.jsonl",
)

print("Done! 🎉")
