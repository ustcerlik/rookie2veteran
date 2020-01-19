# det_datasets

det datasets downloads and uploads

# docker

```bash
docker run -ti -v /ssd:/ssd --rm ${DOCKER_VERSION} bash

cd /workspace/det_datasets/
```

## download demo

```python
python main.py --config_file configs/download_demo.yaml --logfile logs/log.log
```

## upload demo
### upload val data demo
```python
python main.py --config_file configs/upload_valdata_demo.yaml
```

### upload train data using single tar demo (recommended for train data)

This is relativly easy to maintain 
```python
python main.py --config_file configs/upload_single_demo.yaml
```

### upload train data using multi tars demo
```python
python main.py --config_file configs/upload_multi_demo.yaml
```