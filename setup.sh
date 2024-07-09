#!/bin/bash

PYTHON_VERSION="3.11.7"

if ! command -v pyenv &> /dev/null; then
    echo "pyenv não está instalado. Instale-o primeiro."
    exit 1
fi

if ! pyenv versions --bare | grep -q "$PYTHON_VERSION"; then
    echo "Instalando Python $PYTHON_VERSION..."
    pyenv install "$PYTHON_VERSION"
fi

pyenv local "$PYTHON_VERSION"

if ! command -v virtualenv &> /dev/null; then
    echo "virtualenv não está instalado. Instale-o primeiro."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Criando novo ambiente virtual..."
    virtualenv -p $(pyenv which python) venv
fi

source venv/bin/activate

if [ -f "requirements.txt" ]; then
    echo "Instalando dependências..."
    pip install -r requirements.txt
else
    echo "Arquivo requirements.txt não encontrado."
fi

echo "Ambiente configurado com Python $PYTHON_VERSION e ambiente virtual ativado."
