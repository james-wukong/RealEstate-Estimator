FROM python:3.12-slim

WORKDIR /code

RUN apt-get update && apt-get install -y libgomp1 && apt-get clean

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

EXPOSE 8000

# CMD ["fastapi", "run", "app/main.py", "--port", "8000"]