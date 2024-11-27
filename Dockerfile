# Use uma imagem base do Python
FROM python:3.10-slim

# Definir diretório de trabalho no container
WORKDIR /app

# Copiar os arquivos do projeto para o container
COPY app/ /app/

# Instalar dependências
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expor a porta que será usada pela API (apenas para a API)
EXPOSE 5000
