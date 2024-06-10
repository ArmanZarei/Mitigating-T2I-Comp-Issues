#!/bin/bash
#SBATCH --job-name=syngen_t2i_dataset_generation
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --time=24:00:00

# Command to run this: for i in {0..19}; do sbatch --output="T2I-CompBench-dataset/shape/syngen-logs/slurm-logs-$i/out-clip.log" --error="T2I-CompBench-dataset/shape/syngen-logs/slurm-logs-$i/err-clip.log" --export=chunk_idx=$i generate_dataset_using_syngen.sh; done

category="shape"

# Check if chunk_idx variable is defined
if [ -z ${chunk_idx+x} ]; then
    echo "Error: chunk_idx variable is not defined."
    exit 1
fi

# Check if chunk_idx variable is an integer
if ! [[ $chunk_idx =~ ^[0-9]+$ ]]; then
    echo "Error: chunk_idx is not an integer."
    exit 1
fi

# Check if chunk_idx variable is in the valid range
if (( chunk_idx < 0 || chunk_idx > 19 )); then
    echo "Error: Chunk index is not within the valid range."
    exit 1
fi

syngen_repo_dir="path_to_Syntax-Guided-Generation_repo"
cd "$syngen_repo_dir"

eval "$(conda shell.bash hook)"
conda activate syngen

seeds=(691508 336326 282695 354924 203276 646321 887284 83885 619235 699179 287119 120858 413507 935979 887891 64760 565942 814921 40504 133920 172311 940495 911719 810165 237019 464600 859085 931200 402570 301953 181400 162301 72730 518139 31502 769109 668286 380713 127462 790784 404021 60066 421926 440560 776443 496661 105903 548495 710175 618537)

# Read file line by line into an array
readarray -t prompts < "T2I-CompBench-dataset/$category.txt"

# Loop through the array to remove trailing dots
for ((i=0; i<${#prompts[@]}; i++)); do
    prompts[$i]=$(echo "${prompts[$i]}" | sed 's/\.$//')
done

prompts_length=${#prompts[@]}
chunk_size=$(( (prompts_length + 19) / 20 ))
start_index=$(( chunk_idx * chunk_size ))
end_index=$(( (chunk_idx + 1) * chunk_size ))
prompts_chunk=("${prompts[@]:start_index:end_index}")
echo "[Bash Script] Chunk [idx: $chunk_idx] -> $start_index - $end_index"
echo "[Bash Script] first prompt of the chunk: \"${prompts_chunk[0]}\""
echo "[Bash Script] first prompt of the chunk: \"${prompts_chunk[-1]}\""

base_target_dir="T2I-CompBench-dataset/$category"

for prompt in "${prompts_chunk[@]}"; do
    echo "[Bash Script - $(date +'%Y-%m-%d %H:%M:%S')] Starting SynGen run.py for prompt \"$prompt\""

    if [ -d "$base_target_dir/$prompt/syngen/" ] && [ $(ls "$base_target_dir/$prompt/syngen/" -1 | wc -l) -eq ${#seeds[@]} ]; then
        echo "[Bash Script] $(ls "$base_target_dir/$prompt/syngen/" -1 | wc -l) samples have already been generated for this prompt"
        echo "[Bash Script] Skipping this prompt..."
        echo "[Bash Script - $(date +'%Y-%m-%d %H:%M:%S')] Finished SynGen run.py for prompt \"$prompt\""
        continue
    fi

    echo "[Bash Script] Removing files in output_$chunk_idx/ directory"
    rm output_$chunk_idx/*

    python run.py --prompt "$prompt" --output_directory "./output_$chunk_idx" --seeds ${seeds[@]}
    
    rm -r "$base_target_dir/$prompt/syngen"
    mkdir -p "$base_target_dir/$prompt/syngen"
    
    echo "[Bash Script] Copying files from output_$chunk_idx/ to T2I-CompBench-dataset/$category/$prompt/syngen/"
    cp $syngen_repo_dir/output_$chunk_idx/* "$base_target_dir/$prompt/syngen/"

    echo "[Bash Script - $(date +'%Y-%m-%d %H:%M:%S')] Finished SynGen run.py for prompt \"$prompt\""
done
