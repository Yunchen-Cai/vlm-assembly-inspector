VLM Assembly Inspector: Industrial Quality Control with Qwen2.5-VL
Fine-tuning Vision-Language Models for Fine-Grained Industrial Action Recognition

This repository contains the implementation for fine-tuning Qwen2.5-VL-7B-Instruct using QLoRA (4-bit quantization). The project is designed to act as an AI Quality Inspector for industrial assembly lines, capable of recognizing micro-actions (e.g., Grasp, Insert, Tighten), identifying tools/parts, and outputting structured JSON data from single video frames.

✨ Key Features
Industrial Precision: Specialized in distinguishing fine-grained assembly actions often missed by general VLMs.

Structured Output: Fine-tuned to output strict JSON formats (action, tool, part, phase), eliminating hallucinated descriptions.

Efficient Training: Optimized for consumer/workstation GPUs (e.g., NVIDIA RTX A4000/3090) using QLoRA and Flash Attention.

Robust Pipeline: Includes tools for data simplification, pipeline sanity checks, and interactive inference.

📂 Project Structure
Note: Large datasets (data/) and trained model weights (output/, cache/) are excluded from this repository to ensure a lightweight codebase.

Plaintext

vlm-assembly-inspector/
├── src/                        # Source code directory
│   ├── train_lora.py           # Main QLoRA fine-tuning script
│   ├── interactive_inference.py# CLI tool for real-time testing
│   ├── debug_pipeline.py       # Sanity check script (checks data loading/shapes)
│   └── simplify_dataset.py     # Preprocessing tool to clean JSON annotations
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
🚀 Installation
Prerequisites
GPU: NVIDIA GPU with >= 16GB VRAM (e.g., RTX A4000, 3090, 4090).

Driver: CUDA 12.1 or higher recommended.

OS: Windows or Linux.

Setup Steps
Clone the repository:

Bash

git clone https://github.com/Yunchen-Cai/vlm-assembly-inspector.git
cd vlm-assembly-inspector
Install Python dependencies:

Bash

pip install -r requirements.txt
💾 Data Preparation (Important)
Since the dataset is not included in the repo, you must organize your local data as follows:

Create a data folder in the root directory.

Create a temp_frames folder inside data/ and put your images there.

Place your annotation JSON file (e.g., clean_assembly_sft_simple.json) inside data/.

Directory Layout:

Plaintext

vlm-assembly-inspector/
└── data/
    ├── clean_assembly_sft_simple.json
    └── temp_frames/
        ├── 20251212_step1.jpg
        ├── 20251212_step2.jpg
        └── ...
JSON Annotation Format:

JSON

[
  {
    "image": ["temp_frames/20251212_step1.jpg"],
    "conversations": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "temp_frames/20251212_step1.jpg"},
          {"type": "text", "text": "You are an assembly quality inspector..."}
        ]
      },
      {
        "role": "assistant",
        "content": "{\"action\": \"Tighten\", \"tool\": \"Hex Wrench\", \"part\": \"Capscrew\", \"phase\": \"finish\"}"
      }
    ]
  }
]
🛠️ Usage Workflow
1. Data Preprocessing (Optional)
If your dataset contains verbose natural language descriptions (e.g., "Description: The worker is..."), use this script to strip them and keep only the JSON object for better model convergence.

Bash

python src/simplify_dataset.py
2. Pipeline Sanity Check
Highly Recommended: Before starting a long training session, run this script to ensure images are loaded correctly, tensors are not empty, and the prompt format is correct.

Bash

python src/debug_pipeline.py
3. Training
Start the fine-tuning process. The script automatically detects the data/ folder.

Bash

python src/train_lora.py
Configuration: 3 Epochs, Rank 16, Alpha 32, Batch Size 1 (with Gradient Accumulation).

Monitoring: Loss is logged every step. You can use tensorboard --logdir output to visualize.

4. Interactive Inference
Test your trained model interactively. The script loads the base model and your LoRA adapter once, then waits for image filenames.

Bash

python src/interactive_inference.py
Input: Enter the filename of an image (e.g., frame_123.jpg) located in data/temp_frames/.

Output: The model will generate the assembly analysis in JSON format.

⚙️ Technical Details
Base Model: Qwen/Qwen2.5-VL-7B-Instruct

Fine-tuning Method: QLoRA (Quantized Low-Rank Adaptation)

Compute Precision: bfloat16

Quantization: 4-bit nf4 (Normal Float 4) via bitsandbytes

📝 License
This project is licensed under the MIT License.
