# ranzcr-2021

## environment

#### Docker build
```
cd docker
sudo docker build -t ranzcr2021:v0 ./
```

#### Docker run
```
export DATA="/home/yuichi/Desktop/kaggle/RANZCR-2021/data"
sudo docker run -it --rm --name ranzcr0\
 --gpus all --shm-size=100g\
 -v $DATA:/data -v $PWD:/workspace\
 -p 1111:8888 -p 1001:6006 --ip=host\
 ranzcr2021:v1
```