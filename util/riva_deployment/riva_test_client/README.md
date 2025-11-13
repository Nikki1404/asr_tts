# Light weight testing client for Riva ASR

#To build the docker image 
docker build --no-cache -t riva_test_client:1.0.0 .

#for running the container
docker run --rm --network=host -e RIVA_URI=localhost:5000 riva_test_client:1.0.0    