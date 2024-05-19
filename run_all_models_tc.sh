python main.py flan-t5-small -S toxic-chat -D 0 --regression_device cuda:0
python main.py flan-t5-large -S toxic-chat -D 0 --regression_device cuda:0
python main.py flan-t5-xl -S toxic-chat -D 0 --regression_device cuda:0
python main.py tiny-llama -S toxic-chat -D 0 --regression_device cuda:0
python main.py llama-2 -S toxic-chat -D 0 --regression_device cuda:0
python main.py llama-3 -L 2e-4 -R 1e-2 -S toxic-chat -D 0 --regression_device cuda:0
python main.py llama-2-13b -S toxic-chat -D 0,1 --regression_device cuda:0
python main.py mistral-7b -S toxic-chat -D 0 --regression_device cuda:0
python main.py vicuna-7b -S toxic-chat -D 0 --regression_device cuda:0
python main.py gpt2 -S toxic-chat -D 0 --regression_device cuda:0
python main.py llama-2-7b-gptq -S toxic-chat -D 0 --regression_device cuda:0
python main.py llama-2-13b-gptq -S toxic-chat -D 0 --regression_device cuda:0
