# Use uma imagem base do Python
FROM python:3.10-slim

# Definir diretório de trabalho no container
WORKDIR /app

# Copiar os arquivos do projeto para o container
COPY app/ /app/

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta que será usada pela aplicação
EXPOSE 5000

# Comando para iniciar a aplicação
CMD ["python", "app.py"]
