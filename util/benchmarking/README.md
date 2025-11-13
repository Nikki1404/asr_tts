# mistral-voxtral-poc


## Benchmarking doc
- https://exlservicenam-my.sharepoint.com/:x:/g/personal/om224604_exlservice_com/EZ4aa5HAHjdAolpnqHq41CsBX_Yk7rvSsAnIgaPNE8iSvQ?wdOrigin=TEAMS-WEB.p2p_ns.rwc&wdExp=TEAMS-TREATMENT&wdhostclicktime=1753452239413&web=1

## Installation
- https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
- If files not already copied:
    - create folder in the ssh machine as necessary
    - scp -r mistral_voxtral kunal259787@10.90.126.78:/home/CORP/kunal259787
- ssh <ec2 machine ip with GPU>: Example: ssh 10.90.126.78
- export https_proxy="http://163.116.128.80:8080"
- export http_proxy="http://163.116.128.80:8080"
- with transformers:
    - huggingface-cli login
    - Download the model (if not already downloaded):
        - git lfs install
        - git clone https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
    - conda create --name my_env_name python=3.10 (incase of Anaconda)
    - python3 -m venv venv
<<<<<<< Updated upstream
    - source venv/bin/activate
    - pip install --upgrade -r requirements.txt
=======
    - pip install git+https://github.com/huggingface/transformers
    - pip install --upgrade "mistral-common[audio]"
    - pip install accelerate
    - pip install librosa
>>>>>>> Stashed changes
    - python3 test.py
        - In case of memory errors, see which process is taking much memory (using command 'nvidia-smi') and stop them.

## Notes
- Mini model works for ~1 minute chunks nicely
- Sampling rate does not seem to have much effect

## Uses

## Models

## Model Comparisons

## Deployment
