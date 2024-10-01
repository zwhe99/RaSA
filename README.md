##### Environment

```shell
pip3 install -r requirements.txt
pip3 install -e ./peft --config-settings editable_mode=compat
pip3 install -e ./lm-evaluation-harness --config-settings editable_mode=compat
pip3 install -e ./bigcode-evaluation-harness --config-settings editable_mode=compat

```



##### Download training data

```shell
python3 scripts/tools/dl_dataset.py ise-uiuc/Magicoder-Evol-Instruct-110K data/ise-uiuc/Magicoder-Evol-Instruct-110K
python3 scripts/tools/dl_dataset.py meta-math/MetaMathQA data/meta-math/MetaMathQA
```



**Training scripts**

 `scripts/train` provides training scripts for all the main experiments.



##### Evaluatoin

* Humaneval+

  * Generate response

    ```shell
    EXP_DIR=path_to_this_experiment
    accelerate launch --gpu_ids 0 humaneval_eval.py \
        --model $PATH_TO_BASE_MODEL \
        --peft_model $PATH_TO_PEFT_MODEL \
        --tasks humanevalsynthesize-python \
        --prompt alpaca-chat \
        --do_sample True \
        --temperature 0.2 \
        --n_samples 50 \
        --batch_size 10 \
        --max_length 2048 \
        --allow_code_execution \
        --precision bf16 \
        --metric_output_path $EXP_DIR/metric_output_path.json \
        --save_generations \
        --save_generations_path $EXP_DIR/save_generations_path.json \
        --generation_only
    ```

  * Compute pass@1 and pass@10

    ```shell
    python3 humaneval_eval.py \
        --tasks humanevalplus \
        --n_samples 50 \
        --num_workers 48 \
        --timeout 20 \
        --k 1 10 \
        --allow_code_execution \
        --metric_output_path $EXP_DIR/metric_output_path.json \
        --load_generations_path $EXP_DIR/save_generations_path_humanevalsynthesize-python.json \
        --results_path $EXP_DIR/results.json
    ```

    The final result will be saved in `EXP_DIR/results.json`.

* MATH

  ```SHELL
  EXP_DIR=path_to_this_experiment
  python3 math_eval.py \
      --base_model $PATH_TO_BASE_MODEL \
      --chat_template_name alpaca-chat \
      --output_dir $EXP_DIR \
      --data_file data/MATH/MATH_test.jsonl \
      --peft_model $PATH_TO_PEFT_MODEL \
      --bf16 True \
      --batch_size 8
  ```

  The final result will be saved in `EXP_DIR/result.log`.	