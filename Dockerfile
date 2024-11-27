# Dockerfile

# Use uma imagem base do Python (como exemplo)
FROM python:3.9

# Configurações básicas
WORKDIR /app

# Instale as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie o código da aplicação
COPY . .

# Exponha a porta que o app usará
EXPOSE 5000

# Comando para iniciar com Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
