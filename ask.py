from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
from utils.slots import extract_slot, lookup_in_db

def ask(query, model, tokenizer, df):
    # Format prompt
    messages = [{"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate response template (có {so})
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    outputs = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    response = outputs[0]["generated_text"]

    # Trích slot từ query
    slots = extract_slot(query)
    so = lookup_in_db(slots, df)

    # Thay {so} trong response
    final_answer = response.replace("{so}", str(so))
    return final_answer

def main():
    print(">>> Load model đã fine-tune...")
    model_path = "./phi3-finetuned"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load dataframe từ dataset JSONL để lookup
    df = pd.read_json("dataemploy.jsonl", lines=True)

    while True:
        query = input("\nHỏi gì đó (hoặc 'exit'): ")
        if query.lower() == "exit":
            break
        answer = ask(query, model, tokenizer, df)
        print(">>> Trả lời:", answer)

if __name__ == "__main__":
    main()
