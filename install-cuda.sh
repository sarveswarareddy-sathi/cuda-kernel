#!/usr/bin/env bash
set -e

echo "=== Detecting Ubuntu version ==="
UBUNTU_VERSION=$(lsb_release -rs)
echo "Ubuntu version: $UBUNTU_VERSION"

# Map Ubuntu version to NVIDIA repo name
case "$UBUNTU_VERSION" in
    24.04) DISTRO=ubuntu2404 ;;
    22.04) DISTRO=ubuntu2204 ;;
    20.04) DISTRO=ubuntu2004 ;;
    *)
        echo "Unsupported Ubuntu version for this script."
        exit 1
        ;;
esac

echo "=== Removing any old CUDA repo entries ==="
sudo rm -f /etc/apt/sources.list.d/cuda*

echo "=== Adding NVIDIA GPG key and repo for $DISTRO ==="
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/3bf863cc.pub
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/ /" | \
    sudo tee /etc/apt/sources.list.d/cuda.list

echo "=== Updating package lists ==="
sudo apt update

echo "=== Installing latest CUDA Toolkit 12.6 (change version if needed) ==="
sudo apt install -y cuda-toolkit-12-6 build-essential wget

echo "=== Adding CUDA to PATH and LD_LIBRARY_PATH ==="
if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
    {
        echo 'export PATH=/usr/local/cuda/bin:$PATH'
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH'
    } >> ~/.bashrc
fi

echo "=== Reloading shell configuration ==="
source ~/.bashrc

echo "=== Verifying nvcc installation ==="
nvcc --version || { echo "nvcc not found in PATH"; exit 1; }

echo "=== Setup complete ==="
echo "Note: CUDA code will compile here but GPU kernels will not run in Codespaces (no GPU present)."

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

nvcc --version
