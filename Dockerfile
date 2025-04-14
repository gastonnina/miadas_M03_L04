
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install pdm && pdm install
CMD ["pdm", "run", "python", "dashboard.py"]
