@echo off
echo === Construindo projeto (clean + package) ===

REM Executa clean e package silenciosamente
mvn -q clean package

IF %ERRORLEVEL% NEQ 0 (
    echo ERRO: A compilação falhou.
    exit /b 1
)

echo === Executando aplicação ===
java --enable-preview -jar target\ProjetoAlgebraLinear-1.0-SNAPSHOT.jar
