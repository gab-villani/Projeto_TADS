# Use uma imagem base do Python
FROM python:3.10-slim

# Definir diretório de trabalho no container
WORKDIR /app

# Copiar os arquivos do projeto para o container
COPY app/ /app/

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta que será usada pela aplicação# Use uma imagem base do Python
FROM python:3.10-slim

# Definir diretório de trabalho no container
WORKDIR /app

# Copiar os arquivos do projeto para o container
COPY app/ /app/

# Instalar dependências
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expor a porta que será usada pela aplicação
EXPOSE 5000

# Comando para rodar o treinamento e depois iniciar a API
CMD ["bash", "-c", "python init.py && python app.py"]

EXPOSE 5000

# Comando para iniciar a aplicação
CMD ["python", "app.py"]
