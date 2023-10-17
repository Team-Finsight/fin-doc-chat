FROM python:3.9.7-bullseye

WORKDIR /

COPY ./ .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

EXPOSE 8080
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
