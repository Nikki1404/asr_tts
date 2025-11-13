## Steps
- wget --content-disposition
https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.40.0/files/ngccli_linux.zip
-O ngccli_linux.zip && unzip ngccli_linux.zip
- find ngc-cli/ -type f -exec md5sum {} + | LC_ALL=C sort | md5sum -c ngc-cli.md5
- sha256sum ngccli_linux.zip
- chmod u+x ngc-cli/ngc
- echo "export PATH=\"\$PATH:$(pwd)/ngc-cli\"" >> ~/.bash_profile && source ~/.bash_profile
- ngc config set
Then set your api-key and configuration
- ngc registry resource download-version nvidia/riva/riva_quickstart:2.19.0
- cd riva_quickstart_v2.19.0
- Download the folder named rmir from here s3://cx-rnd-ml/riva/rmir/
- Copy the config.sh file inside the above mentioned folder named riva_quickstart_v2.19.0
    - In config.sh update 'riva_model_loc' to the location of the above downloaded rmir folder
- bash riva_init.sh
- bash riva_start.sh

## RIVA Deployment
- [helper cmd] copy files to s3: aws s3 cp rmir s3://cx-rnd-ml/riva --recursiv

- Update the folder path in Dockerfile where the models are downloaded

Then build and run the docker container:
- docker build --no-cache -t riva-exl:2.19.0 .
- In a new gpu machine we need to: apt install nvidia-utils-580
- docker run --gpus all -p 50051:50051 riva-exl:2.19.0

Update the repo-name and version tag:
- docker tag riva-exl:2.19.0 058264113403.dkr.ecr.us-east-1.amazonaws.com/riva-speech:2.19.0

- Add the aws access keys to the enviroment

Login to ECR:
- aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 058264113403.dkr.ecr.us-east-1.amazonaws.com

Finally push the image:
- docker push 058264113403.dkr.ecr.us-east-1.amazonaws.com/riva-speech:2.19.0

- Link: https://us-east-1.console.aws.amazon.com/ecr/repositories/private/058264113403/riva-speech/_/image/sha256:fe4ebf6472c4f401dc61d8b39913c07f8c4082f244e0ebd940193db6922fe568/details?region=us-east-1

## Share fine-tuned RIVA model
- copy Dockerfile to some #location# in 10.90.126.61 machine for example
- cd #location#
- cp -r /home/CORP/RIVA/riva_models_19_stable/models/ ./
- docker build -t riva_finetuned_inspira .
- docker save -o riva_finetuned_inspira.tar riva_finetuned_inspira:latest
    - To load it: docker load -i riva_finetuned_inspira.tar
    - run it with the docker run command mentioned above