

### Docker

**download-link**: https://pan.baidu.com/s/1HLqJ52xJRo6ucLzToIEkBw?pwd=1024

```
docker load -i danceba.tar
docker images
docker run -i danceba:latest /bin/bash
```


### Debug
```
# ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /root/anaconda3/lib/python3.9/site-packages/scipy/optimize/_highs/_highs_wrapper.cpython-39-x86_64-linux-gnu.so)
export LD_LIBRARY_PATH=/root/anaconda3/envs/danceba/lib:$LD_LIBRARY_PATH

# ModuleNotFoundError: No module named 'torch'
conda deactivate base
conda activate danceba

# ImportError: libGL.so.1: cannot open shared object file: No such file or directory
pip install opencv-python-headless
pip install numpy==1.26
```
