# AI SOP Monitor — common commands.
#
# Works with GNU Make on Linux/macOS (sh) AND on Windows from PowerShell/cmd:
# recipes avoid POSIX-only syntax (env-var prefixes, rm/mkdir), so no Git
# Bash is required. If `uv` is installed it is preferred for creating .venv.
#
# Override knobs:
#   make install CUDA_INDEX=https://download.pytorch.org/whl/cu121
#   make run PORT=9000

# ---------- platform detection ----------------------------------------------
ifeq ($(OS),Windows_NT)
  VENV_BIN := .venv/Scripts
  EXE      := .exe
else
  VENV_BIN := .venv/bin
  EXE      :=
endif

PY          := $(VENV_BIN)/python$(EXE)
SOP_MONITOR := $(VENV_BIN)/sop-monitor$(EXE)

# ---------- knobs (override on the command line) ----------------------------
CUDA_INDEX ?= https://download.pytorch.org/whl/cu126
PORT       ?= 8000
IMAGE      ?= sop-monitor

# ---------- meta ------------------------------------------------------------
.DEFAULT_GOAL := help
.PHONY: help venv install install-cpu gpu-check run dev \
        docker-build docker-run clean clean-venv

help: ## Show this help.
	@echo "AI SOP Monitor targets:"
	@echo "  make venv            Create .venv"
	@echo "  make install         Editable install with CUDA 12.x torch wheels"
	@echo "  make install-cpu     Editable install with CPU-only torch (also for macOS)"
	@echo "  make gpu-check       Print torch / CUDA / device info"
	@echo "  --"
	@echo "  make run             Start the monitor at http://127.0.0.1:$(PORT)"
	@echo "  make dev             Start with auto-reload (development)"
	@echo "  --"
	@echo "  make docker-build    Build the CPU Docker image ($(IMAGE))"
	@echo "  make docker-run      Run the image on :$(PORT) with a persistent data volume"
	@echo "  --"
	@echo "  make clean           Empty monitor_data/ (uploads + results)"
	@echo "  make clean-venv      Delete .venv (force-recreate)"

# ---------- environment -----------------------------------------------------
venv: $(PY) ## Create the virtualenv (uv if available, --seed adds pip).
$(PY):
	uv venv --seed || py -3 -m venv .venv || python3 -m venv .venv || python -m venv .venv

install: venv ## Editable install with CUDA torch wheels.
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e .[gpu] --extra-index-url $(CUDA_INDEX)

install-cpu: venv ## Editable install with CPU-only torch.
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e .[cpu]

gpu-check: ## Print model / torch device / SOP info.
	$(SOP_MONITOR) --check

# ---------- run --------------------------------------------------------------
run: ## Start the SOP monitor.
	$(SOP_MONITOR) --port $(PORT)

dev: ## Development server with auto-reload.
	$(PY) -m uvicorn sop_monitor.app:app --reload --port $(PORT)

# ---------- docker -----------------------------------------------------------
docker-build: ## Build the CPU container image.
	docker build -t $(IMAGE) .

docker-run: docker-build ## Run the container; results persist in the sop_data volume.
	docker run --rm -p $(PORT):8000 -v sop_data:/app/monitor_data $(IMAGE)

# ---------- cleanup ---------------------------------------------------------
# File ops go through Python (any interpreter on PATH) instead of rm/mkdir so
# they work from cmd/PowerShell as well as sh.
clean: ## Empty monitor_data (uploads + processed results).
	py -3 -c "import shutil; shutil.rmtree('monitor_data', ignore_errors=True)" || python3 -c "import shutil; shutil.rmtree('monitor_data', ignore_errors=True)" || python -c "import shutil; shutil.rmtree('monitor_data', ignore_errors=True)"

clean-venv: ## Nuke .venv (force-recreate next install).
	py -3 -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)" || python3 -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)" || python -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
