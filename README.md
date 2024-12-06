# LLaVA Video Captioner

This tool uses LLaVA (Large Language and Vision Assistant) to generate detailed descriptions of video content, with support for batch processing multiple videos.

## Features
- Batch video processing from a specified folder
- Frame sampling with configurable FPS
- Memory-optimized processing with batch support
- Customizable prompt templates
- CSV output for generated captions
- 4-bit or 8-bit model quantization support

## Requirements
- Python 3.10 or higher
- NVIDIA GPU with CUDA support
- Sufficient GPU memory (minimum 8GB recommended)

## Installation_

1. Clone the repository:
```bash
git clone https://github.com/cseti007/LLaVA-Video-7B-Qwen2-captioner.git
cd llava-video-captioner
```

2. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

3. Install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

4. Install the LLaVA repository:
```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e .
cd ..
```

5. Install additional dependencies:
```bash
pip install decord transformers bitsandbytes
pip install einops av open_clip_torch
pip install 'accelerate>=0.26.0'
```

Note: If you encounter SSL certificate errors during installation, you can use:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org [package_name]
```

## Configuration

Edit the `CONFIG` dictionary in the script to customize:
- Input/output paths
- Video processing parameters
- Model settings
- Generation parameters
- Prompt template

Example configuration:
```python
CONFIG = {
    'max_frames_num': 24,       # Number of frames to process
    'fps': 1.0,                 # Sampling frequency
    'force_sample': True,       # Force uniform sampling
    'batch_size': 8,           # Batch size for frame processing
    'input_folder': "path/to/videos",
    'output_csv': "path/to/output.csv",
    
    # Model parameters
    'model_path': "lmms-lab/LLaVA-Video-7B-Qwen2",
    'model_name': "llava_qwen",
    'quantization_bits': 4,     # 4 or 8 bit quantization
    
    # Generation parameters
    'max_new_tokens': 512,
    'temperature': 0.2,
    'do_sample': True,
}
```

## Usage

1. Update the configuration in the script:
   - Set `input_folder` to your videos directory
   - Set `output_csv` to your desired output location

2. Run the script:
```bash
python llava_video-video-captioner.py
```

The script will:
1. Process all videos in the input folder
2. Generate descriptions using the specified prompt template
3. Save results to the specified CSV file

## Supported Video Formats
- MP4
- AVI
- MOV
- MKV

## Memory Optimization

The script includes several memory optimization features:
- Batch processing of frames
- Garbage collection
- CUDA memory management
- 4-bit quantization (default)

You can adjust these parameters in the CONFIG dictionary:
- `batch_size`: Number of frames processed at once
- `max_frames_num`: Total frames to sample from each video
- `quantization_bits`: 4 or 8-bit quantization

## Troubleshooting

Common issues:

1. CUDA out of memory:
   - Reduce `batch_size`
   - Reduce `max_frames_num`
   - Use 4-bit quantization
   - Ensure no other processes are using GPU memory

2. Slow processing:
   - Increase `batch_size` if memory allows
   - Adjust `fps` for fewer frames
   - Consider using 8-bit quantization for better speed

3. Installation issues:
   - Make sure CUDA is properly installed
   - Check Python version compatibility
   - Try installing dependencies one by one to identify issues

4. Import errors:
   - Ensure all dependencies are installed
   - Check if virtual environment is activated
   - Verify CUDA compatibility with PyTorch version

## Acknowledgments
- LLaVA Team for the original model and implementation
- [Other acknowledgments]
#
