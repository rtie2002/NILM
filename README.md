# NILM

## Installation

### 1. Install UV (Fast Python Package Installer)

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Start Virtual Environment

```powershell
cd 'c:\Users\Raymond Tie\Desktop\NILM'
.\venv\Scripts\activate
```

### 3. Install nilmtk packages

```bash
uv pip install git+https://github.com/nilmtk/nilmtk.git
uv pip install git+https://github.com/nilmtk/nilmtk-contrib.git
```

**Note:** The nilmtk-contrib package requires Python 3.11.5 exactly.

### 4. Stop Virtual Environment

```powershell
deactivate
```

## Running

cursor-notebook-mcp --transport streamable-http --allow-root "C:\Users\Raymond Tie\Desktop\NILM" --host 127.0.0.1 --port 8080
