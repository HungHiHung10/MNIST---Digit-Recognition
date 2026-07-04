FROM python:3.9-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user ./requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r $HOME/app/requirements.txt

COPY --chown=user . $HOME/app

EXPOSE 7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "run:app"]
