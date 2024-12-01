import os
from tqdm import tqdm

# Define configurations
conf_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
conf_methods = ["softmax_diff", "softmax_max", "state"]
datasets_table3 = ["cnn_dm_summarization", "xsum_summarization", "human_eval"]
generation_strategies = ["self_speculative", "autoregressive", "autoregressive_with_early_exit"]
models_table3 = ["facebook/layerskip-llama2-7B", "facebook/layerskip-llama2-13B"]

# Table 5 specific configurations
table5_model = "facebook/layerskip-codellama-7B"
table5_dataset = "human_eval"
table5_generation_strategies = ["autoregressive", "self_speculative"]
table5_exit_layer = 6

output_dir = "./logs"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Base command template
base_command = (
    "torchrun benchmark.py "
    "--num_samples 100 "
    "--sample False "
    "--output_dir {output_dir} "
)

# Calculate total tasks for Table 3
total_table3_tasks = len(models_table3) * len(datasets_table3) * len(generation_strategies)
total_table3_tasks += len(conf_thresholds) * len(conf_methods) * len(models_table3) * len(datasets_table3)  # For autoregressive_with_early_exit

# Create a tqdm progress bar
task_count = 0  # To track completed tasks for progress
with tqdm(total=total_table3_tasks, desc="Running Table 3 tasks", unit="task") as pbar:
    # Iterate over Table 3 configurations
    for model in models_table3:
        for dataset in datasets_table3:
            # Adjust n_shot based on the dataset
            n_shot = 1 if dataset == "cnn_dm_summarization" else 0
            for generation_strategy in generation_strategies:
                if generation_strategy == "autoregressive_with_early_exit":
                    # Loop through conf_thresholds and conf_methods for "autoregressive_with_early_exit"
                    for conf_threshold in conf_thresholds:
                        for conf_method in conf_methods:
                            command = (
                                base_command +
                                f"--model {model} "
                                f"--dataset {dataset} "
                                f"--n_shot {n_shot} "
                                f"--generation_strategy {generation_strategy} "
                                f"--conf_threshold {conf_threshold} "
                                f"--conf_method {conf_method} "
                                f"--exit_layer 8 "
                            )
                            output_file = os.path.join(
                                output_dir,
                                f"results_table3_{model.replace('/', '_')}_{dataset}_{generation_strategy}_{conf_threshold}_{conf_method}.json"
                            )
                            command += f" --output_file {output_file}"
                            print(f"Running: {command}")
                            os.system(command)

                            # Update progress bar
                            pbar.update(1)

                else:  # For "self_speculative" and "autoregressive", skip conf_thresholds and conf_methods
                    command = (
                        base_command +
                        f"--model {model} "
                        f"--dataset {dataset} "
                        f"--n_shot {n_shot} "
                        f"--generation_strategy {generation_strategy} "
                        f"--exit_layer 8 "
                    )
                    if generation_strategy == "self_speculative":
                        command += "--num_speculations 12 "
                    output_file = os.path.join(
                        output_dir,
                        f"results_table3_{model.replace('/', '_')}_{dataset}_{generation_strategy}.json"
                    )
                    command += f" --output_file {output_file}"
                    print(f"Running: {command}")
                    os.system(command)

                    # Update progress bar
                    pbar.update(1)

# Calculate total tasks for Table 5 (only for 7B model and human_eval dataset)
total_table5_tasks = len(table5_generation_strategies)

# Create a tqdm progress bar for Table 5 tasks
with tqdm(total=total_table5_tasks, desc="Running Table 5 tasks", unit="task") as pbar:
    # Iterate over Table 5 configurations (only for 7B model and human_eval dataset)
    for generation_strategy in table5_generation_strategies:
        n_shot = 0
        if generation_strategy == "self_speculative":
            command = (
                base_command +
                f"--model {table5_model} "
                f"--dataset {table5_dataset} "
                f"--n_shot {n_shot} "
                f"--generation_strategy {generation_strategy} "
                f"--exit_layer {table5_exit_layer} "
                f"--num_speculations 12 "
            )
        else:  # For "autoregressive"
            command = (
                base_command +
                f"--model {table5_model} "
                f"--dataset {table5_dataset} "
                f"--n_shot {n_shot} "
                f"--generation_strategy {generation_strategy} "
                f"--exit_layer {table5_exit_layer} "
            )
        output_file = os.path.join(
            output_dir,
            f"results_table5_{table5_model.replace('/', '_')}_{table5_dataset}_{generation_strategy}.json"
        )
        command += f" --output_file {output_file}"
        print(f"Running: {command}")
        os.system(command)

        # Update progress bar
        pbar.update(1)
