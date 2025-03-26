.PHONY: *

VENV=/opt/python_venvs/research
PYTHON=$(VENV)/bin/python3
PIP=$(VENV)/bin/pip
MIM=$(VENV)/bin/mim

# You need to carefully check the version of mmcv, torch and cuda here: https://mmcv.readthedocs.io/en/latest/get_started/installation.html


# ================== LOCAL WORKSPACE SETUP ==================
venv:
	~/.pyenv/versions/3.9.17/bin/python -m venv $(VENV)
	@echo 'Path to Python executable $(shell pwd)/$(PYTHON)'

install: venv
	@echo "=== Installing common dependencies ==="
	$(PIP) install --upgrade pip
	$(PIP) install -U torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
	$(PIP) install -r requirements.txt
	$(PIP) install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
	$(PIP) install mmengine mmdet


# ========================= LINTERS ========================
lint:
	$(VENV)/bin/nbstripout examples/*.ipynb