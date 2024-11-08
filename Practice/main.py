from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
	model_name = "gpt2"
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name)
	
	input_text = "Once upon a time,there was a kid"
	
	inputs = tokenizer(input_text, return_tensors="pt")
	outputs = model.generate(**inputs, max_length=80, do_sample=True, top_p=0.95, temperature=0.7)
	generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
	
	print(generated_text)
