{ pkgs, lib, ... }:

{
  # Python stack with uv package manager & writable virtualenv
  languages.python = {
    enable = true;
    venv.enable = true;
    uv.enable = true;
  };

  # Native libraries & build tools for Python wheels (PyTorch, sentencepiece, etc.)
  packages = with pkgs; [
    ruff
    git
    jq
    stdenv.cc.cc.lib
    zlib
  ];

  enterShell = ''
    # WSL2 GPU Acceleration: make Windows-provided CUDA driver shim available to PyTorch and CUDA runtimes
    if [ -d "/usr/lib/wsl/lib" ]; then
      export LD_LIBRARY_PATH="/usr/lib/wsl/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      if [[ ":$PATH:" != *":/usr/lib/wsl/lib:"* ]]; then
        export PATH="/usr/lib/wsl/lib:$PATH"
      fi
    fi

    # Ensure virtualenv activate script preserves WSL2 CUDA library path when sourced outside devenv
    if [ -n "''${VIRTUAL_ENV:-}" ] && [ -f "$VIRTUAL_ENV/bin/activate" ] && [ -d "/usr/lib/wsl/lib" ]; then
      if ! grep -q "/usr/lib/wsl/lib" "$VIRTUAL_ENV/bin/activate"; then
        echo "" >> "$VIRTUAL_ENV/bin/activate"
        echo "# WSL2 GPU Acceleration: expose Windows NVIDIA CUDA driver shims" >> "$VIRTUAL_ENV/bin/activate"
        echo 'if [ -d "/usr/lib/wsl/lib" ]; then' >> "$VIRTUAL_ENV/bin/activate"
        echo '    export LD_LIBRARY_PATH="/usr/lib/wsl/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' >> "$VIRTUAL_ENV/bin/activate"
        echo 'fi' >> "$VIRTUAL_ENV/bin/activate"
      fi
    fi

    # Ensure standard pip executable is present in virtualenv for notebook compatibility
    if [ ! -f "$VIRTUAL_ENV/bin/pip" ]; then
      uv pip install pip --quiet 2>/dev/null || true
    fi

    # Auto-sync dependencies if requirements.txt exists, or display helpful best-practice guidance
    if [ -f requirements.txt ]; then
      echo "📦 Syncing dependencies from requirements.txt via uv..."
      uv pip install -r requirements.txt
    elif [ -f pyproject.toml ]; then
      echo "📦 Syncing project dependencies from pyproject.toml via uv..."
      uv sync
    else
      echo "⚠️  Warning: No 'requirements.txt' found."
      echo "   💡 Best practice is to declare your Python dependencies in 'requirements.txt'."
      echo "   Whenever you create one, direnv will automatically detect and install them via uv!"
    fi

    echo "🐍 Python $(python3 --version 2>/dev/null) | uv ready!"
    echo "💡 Virtualenv active at: $VIRTUAL_ENV"
    echo "   • Fast pip: 'uv pip install <package>' (or standard 'pip install')"
    echo "   • Sync requirements: Auto-managed on save via direnv"
  '';
}
