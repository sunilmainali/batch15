# import os
# from dotenv import load_dotenv
# load_dotenv()
# hf_token = os.getenv("HF_TOKEN")
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
# tokens = tokenizer("What is the capital of France?", return_tensors="pt")
# print(tokens)

# from sentence_transformers import SentenceTransformer
 
# model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
 
# sentences = [
#     "The weather is lovely today.",
#     "It's so sunny outside!",
#     "He drove to the stadium."
# ]
# embeddings = model.encode(sentences)
 
# similarities = model.similarity(embeddings, embeddings)
# print(similarities.shape)
# print(len(embeddings))
# print(similarities)


from transformers import AutoTokenizer, AutoModelForCausalLM
 
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
messages = [
    {"role": "user", "content": "tell me about nepal?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)
 
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


