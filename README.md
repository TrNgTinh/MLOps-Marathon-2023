# MLOps Marathon 2023 - Sample solution

This repository is the sample solution for MLOps Marathon 2023.

## Quickstart

1.  Prepare environment

    ```bash
    # Install python 3.9
    # Install docker version 20.10.17
    # Install docker-compose version v2.6.1
    
    # Here are some steps on how to set up Conda Environment, Docker, and Docker-compose in Linux. If you have already installed them, you can ignore this.

    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
    bash Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
    conda create -n myenv python=3.9
    conda activate myenv
    sudo apt update
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    sudo apt-get install docker-compose-plugin
    sudo apt install docker-compose
    sudo usermod -aG docker ubuntu
    sudo chmod 666 /var/run/docker.sock
    cat /etc/group | grep docker

    # Pip instaal Requirements
    pip install -r requirements.txt
    make mlflow_up
    ```

2.  Train model

    -   Download data, `./data/raw_data` dir should look like. Link Download Data https://drive.google.com/drive/folders/1c0ThOCB_nO_jjuZa_bLghy65HfzMbiXh

        ```bash
        data/raw_data
        ├── .gitkeep
        ├── phase-2
        └── phase-3
            └── prob-1
                ├── features_config.json
                └── raw_train.parquet
        ```

    -   Static data  

        This script runs the select_features function to select the best features of the data. The output of the function is saved to the file data/train_data/phase-id/prob-id/selected_features.pickle. Additionally, the script also trains some models for encoding categorical data and saves the base score of a LightGBM model to the file score.txt.

        ```bash
        python src/static_data.py --phase-id phase-3 --prob-id prob-1
        ```
    
    -   Process data
        
        
        ```bash
        python src/raw_data_processor.py --phase-id phase-3 --prob-id prob-1
        ```

    -   After processing data, `./data/train_data` dir should look like

        ```bash
        data/train_data
        ├── .gitkeep
        └── phase-3
            └── prob-1
                ├── category_index.pickle
                ├── test_x.parquet
                ├── test_y.parquet
                ├── train_x.parquet
                └── train_y.parquet
        ```

    -   Train model

        ```bash
        export MLFLOW_TRACKING_URI=http://localhost:5000
        python src/model_trainer.py --phase-id phase-3 --prob-id prob-1
        ```

    -   Register model: Go to mlflow UI at <http://localhost:5000> and register a new model named **phase-3_prob-1_model-1**

3.  Deploy model predictor

    -   Create model config at `data/model_config/phase-3/prob-1/model-1.yaml` with content:

        ```yaml
        phase_id: "phase-3"
        prob_id: "prob-1"
        model_name: "phase-3_prob-1_model-1"
        model_version: "1"
        ```

    -   Test model predictor

        ```bash
        # run model predictor
        export MLFLOW_TRACKING_URI=http://localhost:5000

        python src/model_predictor.py --model1-config-path data/model_config/phase-3/prob-1/model-1.yaml --model2-config-path data/model_config/phase-3/prob-2/model-1.yaml --port 5040


        # curl in another terminal
        curl -X POST http://localhost:5040/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json

        curl -X POST http://localhost:5040/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json

        curl -X POST http://localhost:5040/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json

        curl -X POST http://localhost:5040/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-0.json

        curl -X POST http://localhost:5040/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-0.json


        # curl in another machine to aws server ( incase using aws server)

        curl -X POST http://ec2-13-250-39-138.ap-southeast-1.compute.amazonaws.com:5040/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json

        curl -X POST http://ec2-54-169-18-28.ap-southeast-1.compute.amazonaws.com:5040/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json

        

        # stop the predictor above
        ```

    -   Deploy model predictor

        ```bash
        make predictor_up
        make predictor_curl
        ```

    -   After running `make predictor_curl` to send requests to the server, `./data/captured_data` dir should look like:

        ```bash
         data/captured_data
         ├── .gitkeep 
         └── phase-3
             └── prob-1
                 ├── {id1}.parquet
                 └── {id2}.parquet
        ```

4.  Improve model

    -   The technique to improve model by using the prediction data is described in `improve_model.md`.
    -   Label the captured data, taking around 3 minutes

        ```bash
        python src/label_captured_data.py --phase-id phase-3 --prob-id prob-1
        ```

    -   After label the captured data, `./data/captured_data` dir should look like:

        ```bash
        data/captured_data
        ├── .gitkeep
        └── phase-3
            └── prob-1
                ├── {id1}.parquet
                ├── {id2}.parquet
                └── processed
                    ├── captured_x.parquet
                    └── uncertain_y.parquet
        ```

    -   Improve model with updated data

        ```bash
        export MLFLOW_TRACKING_URI=http://localhost:5000
        python src/model_trainer.py --phase-id phase-3 --prob-id prob-1 --add-captured-data true
        ```

    -   Register model: Go to mlflow UI at <http://localhost:5000> and register model using the existing name **phase-3_prob-1_model-1**. The latest model version now should be `2`.

    -   Update model config at `data/model_config/phase-3/prob-1/model-1.yaml` to:

        ```yaml
        phase_id: "phase-3"
        prob_id: "prob-1"
        model_name: "phase-3_prob-1_model-1"
        model_version: "2"
        ```

    -   Deploy new model version

        ```bash
        make predictor_restart
        make predictor_curl
        ```

5.  Teardown

    ```bash
    make teardown
    ```
