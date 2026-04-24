run:
    @echo "Running Feature Pipeline"
    @python src/micro_ml/entrypoints/training_endpoint.py

feature:
    @echo "Running Feature Pipeline"
    @python src/micro_ml/entrypoints/training_endpoint.py training.pipeline=disable

train:
    @echo "Running Training Pipeline"
    @python src/micro_ml/entrypoints/training_endpoint.py feature.pipeline=disable