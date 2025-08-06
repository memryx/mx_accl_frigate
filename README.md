# MemryX Python Runtime API for Frigate

This repository contains the **Python runtime API source code** used in the MemryX acceleration module for [Frigate](https://github.com/blakeblackshear/frigate) — NVR With Realtime Object Detection for IP Cameras.

> ⚠️ This repository is **runtime-focused only** and tailored specifically for integration with Frigate Docker deployments.

---

## Getting Started (Standalone Testing)

You can run test script in isolation using a Python virtual environment:

### 1. Clone the Repository

```bash
git clone git@github.com:memryx/mx_accl_frigate.git
```

### 2. Create and Activate a Virtual Environment

```bash
python3.10 -m venv mx_accl
source mx_accl/bin/activate
```

### 3. Link the memryx Module into Site-Packages
Replace the path below with the absolute path to your cloned mx_accl_frigate repo.

```bash
cd mx_accl/lib64/python3.10/site-packages/
ln -s /absolute/path/to/mx_accl_frigate/memryx .
cd 
```

### 4. Install Dependencies
Install all necessary Python packages listed in the freeze file:

```
cd mx_accl_frigate/
pip install -r freeze
```

