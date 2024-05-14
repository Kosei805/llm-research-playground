from llama_cpp import Llama

model_path = 'models/DataPilot-ArrowPro-7B-KUJIRA-IQ4_NL.gguf'

llm = Llama(
  model_path=model_path,
  n_gpu_layers=-1,
)

sys_msg = "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。"

user_query = input(':')
prompt = """[INST] <<SYS>>
{}
<</SYS>>

{}[/INST]
""".format(sys_msg,user_query)

output = llm(
  prompt=prompt,
  max_tokens=512,
  stop=["[INST]","\n"],
  echo=True,
)

print(output)