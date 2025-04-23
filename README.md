# rag_esg


## Data Processing

#### Sub-Target Definitions

We take each SDG and use the sub-goals or sub-targets for each goal and define them in `data/targets.json`

The original datasource is taken form the internal Refinitiv Database and saved as `data/maps.csv`, which is then processed to get the sub-target definitions by running 

```
python3 src/data_processing/extract_targets.py
```

#### Setup Weaviate

Configure the docker yml file and run the following command to setup a weaviate instance in docker

```
docker-compose up -d
```

