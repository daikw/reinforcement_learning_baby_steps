# reinforcement_learning_baby_steps

「Python で学ぶ強化学習」を動かしてみる

```sh
# upgrade pipenv
pip install --upgrade pip
pip install pipenv

# install requirements
pipenv --python 3.10
pipenv install gym jupyter numpy pandas scipy scikit-learn matplotlib tensorflow h5py pygame tqdm
```

# GPU setup

see also: https://gist.github.com/hiraksarkar/b4aff12ccb0f1f1a7cb301f365892f6a

## version compatibility

see: https://www.tensorflow.org/install/source

- Version: tensorflow-2.11.0
- Python version: 3.7-3.10
- Compiler: GCC 9.3.1
- Build tools: Bazel 5.3.0
- cuDNN: 8.1
- CUDA: 11.2

## cuda installation

https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

## cudnn installation

https://developer.nvidia.com/rdp/cudnn-archive

```
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb

cd /var/cudnn-local-repo-ubuntu2204-8.6.0.163
sudo dpkg -i libcudnn8_8.6.0.163-1+cuda11.8_amd64.deb
sudo dpkg -i libcudnn8-dev_8.6.0.163-1+cuda11.8_amd64.deb
sudo dpkg -i libcudnn8-samples_8.6.0.163-1+cuda11.8_amd64.deb
```
