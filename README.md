# NVIDIA AIStore Minimal Cluster Deployment

This repository contains the configuration to deploy a minimal NVIDIA AIS cluster using Docker Compose

## Prerequisites

- Docker installed on your machine
- Docker Compose installed

## Getting Started

### Clone the Repository

```bash
git clone git@github.com:popperony/aistore-sample.git
cd aistore-sample
```

## Configuration

Create a .env file in the root directory of the repository. Use the following as an example:


### .env
```
AIS_BACKEND_PROVIDERS=your_backend_providers
ACCESS_KEY=your_access_key
SECRET_KEY=your_secret_key
ENDPOINT=your_endpoint
AIS_DISK=/path/to/local/disk
```


## Deploy the Cluster
To start the AIS cluster, run the following command:
```
docker-compose up
```

This will start the cluster and map the necessary ports and volumes

## Accessing the Cluster

The AIS cluster will be accessible at `http://localhost:51080`
