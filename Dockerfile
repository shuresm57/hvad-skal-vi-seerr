FROM python:3.12-slim

WORKDIR /app

COPY requirements-infer.txt .
RUN pip install --no-cache-dir -r requirements-infer.txt

COPY export_history.py train_local.py recommend.py ./

# Mount the repo over /app at runtime (-v $(pwd):/app) so library.csv, models/,
# and metrics_history.csv persist on the host between runs instead of
# vanishing with the container.
ENTRYPOINT ["sh", "-c", "python export_history.py && python train_local.py && python recommend.py"]
