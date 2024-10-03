**This directory holds all the training scripts for main experiments.**



The naming of the scripts follows this format:

`llm.{BASE_MODEL}+data.{DATA}+peft.{PEFT}+r.{R}.sh`



**BASE_MODEL**

* `llama-3_1-8b`: meta-llama/Meta-Llama-3.1-8B
* `mistral-3-7b`: mistralai/Mistral-7B-v0.3



**DATA**

* `code`: ise-uiuc/Magicoder-Evol-Instruct-110K
* `math`: meta-math/MetaMathQA



**PEFT**

* `lora`
* `rasa`
* `mora`
* `vera`



**R**

* 8
* 16
* 32
* 1024