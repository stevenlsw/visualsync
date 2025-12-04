conda create -n visualsync python=3.10
conda activate visualsync
pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

cd Tracking-Anything-with-DEVA
pip install -e .
cd ..

cd Grounded-SAM-2
pip install -e .
pip install --no-build-isolation -e grounding_dino
pip install openai imageio dotenv transformers einops imgcat
pip install imageio[ffmpeg]