py -3 -m venv .venv-cu121
if ($LASTEXITCODE -ne 0) { exit 1 }
$p = ".\.venv-cu121\Scripts\python.exe"
& $p -m pip install --quiet --upgrade pip
& $p -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { exit 1 }
& $p -m pip install --quiet numpy==1.26.4 "pillow>=10" "opencv-python-headless>=4.9" simple-lama-inpainting==0.1.2 "ultralytics>=8.2" "huggingface_hub>=0.24" "transformers==4.44.2" "diffusers==0.30.3" "accelerate>=0.33" "safetensors>=0.4" realesrgan==0.3.0 basicsr==1.4.2 "fastapi>=0.111" "uvicorn[standard]>=0.30" python-multipart python-dotenv
& $p -c "import torch; print('VENV READY torch', torch.__version__, 'cuda', torch.cuda.is_available())"
