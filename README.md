**Anonymized Research Repository of Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks**

**All experiments were conducted on:**
- OS: Ubuntu 18.04
- Python: 3.7.16
- CUDA: 11.3
- torch: 1.12.0
- numpy: 1.21.5

**Execution Instructions**
1. init_criterion.py: Initialize criterion.
2. main.py: Train model.
3. inference.py: Test model. You can use our pretrained checkpoint in pretrained_surrogate_results/default, after unzip model.zip.

**Setup**
1. Create conda env
conda create -n nslpfn python=3.7
conda activate nslpfn
2. Install pytorch
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
3. Install other requirements
pip install -r requirements.txt
4-1. Download checkpoint
wget https://huggingface.co/dwlee00/nslpfn/resolve/main/model.pt
4-2. Or, training from scratch
python init_criterion.py --exp_name [exp_name]
python main.py --exp_name [exp_name]
5. Visualize & log NSL-PFN on benchmarks
python inference.py --checkpoint_dir ./pretrained_surrogate_results/[exp_name]