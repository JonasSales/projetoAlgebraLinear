#!/bin/bash

echo "=== Construindo projeto (clean + package) ==="

# Executa clean e package com logs silenciosos
mvn -q clean package

if [ $? -ne 0 ]; then
    echo "ERRO: A compilação falhou."
    exit 1
fi

echo "=== Executando aplicação ==="

java --enable-preview -jar ./target/ProjetoAlgebraLinear-1.0-SNAPSHOT.jar
