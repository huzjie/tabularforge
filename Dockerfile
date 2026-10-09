# tabularforge — tabular foundation model framework (zero-dependency runtime)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY tabularforge ./tabularforge

EXPOSE 8000

# The framework is zero-dependency: no pip install required to run.
# If you install the package, `pip install -e .` also works.
CMD ["python", "-m", "tabularforge", "serve", "--host", "0.0.0.0", "--port", "8000"]
