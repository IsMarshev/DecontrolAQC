# DecontrolAQC



## Doker install 
1. Clone git
```
git clone IsMarshev/DecontrolAQC
```
2. Bild docker from Dokerfile
```
docker build -t decontolaqc/decontrolaqc:latest ./Dockerfile \
```

3.Run it
```
docker run -dp 5000 decontolaqc/decontrolaqc:latest
```