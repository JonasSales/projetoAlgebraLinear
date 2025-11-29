# 🕵️‍♂️ Sistema de Reconhecimento Facial - Fisherfaces

![Java](https://img.shields.io/badge/Java-21-orange?style=for-the-badge&logo=java)
![Maven](https://img.shields.io/badge/Maven-Build-blue?style=for-the-badge&logo=apache-maven)
![Status](https://img.shields.io/badge/Status-Academic%20Demo-green?style=for-the-badge)

Este projeto é uma **demonstração acadêmica** de um sistema de reconhecimento facial implementado em Java. O sistema utiliza conceitos fundamentais de **Álgebra Linear** para identificar indivíduos, combinando técnicas de **PCA (Eigenfaces)** e **LDA (Fisherfaces)**.

Diferente de soluções baseadas em redes neurais profundas (Deep Learning), este projeto foca na implementação matemática explícita de algoritmos de subespaço para classificação de faces.

---

## 🧠 Conceitos Matemáticos Abordados

O núcleo do reconhecimento (`FisherfacesModel`) utiliza a biblioteca `commons-math3` para operações matriciais, implementando:

1.  **Processamento de Imagem**: Conversão para escala de cinza, redimensionamento para 100x100 pixels e equalização de histograma para normalização de iluminação.
2.  **PCA (Análise de Componentes Principais)**: Redução de dimensionalidade focada na variância global dos dados (geração de *Eigenfaces*).
3.  **LDA (Análise Discriminante Linear)**: Projeção que maximiza a distância entre classes (pessoas diferentes) e minimiza a variância intraclasse (mesma pessoa).
4.  **Classificação**: Utilização da *Distância Euclidiana* no espaço projetado para identificar a similaridade entre faces.

---

## 🛠️ Tecnologias Utilizadas

* **Java 21** (com recursos de *Preview* habilitados).
* **Apache Maven** (Gerenciamento de dependências e build).
* **Apache Commons Math 3.6.1** (Álgebra Linear).
* **Java AWT/ImageIO** (Manipulação nativa de imagens).

---

## 📂 Estrutura do Projeto

A estrutura de diretórios esperada para o funcionamento correto do carregador de dados (`DatabaseLoader`) é a seguinte:

```text
├── data
│   ├── database_criminosos
│   │   ├── Individuo_A
│   │   │   └── test.png
│   │   ├── Individuo_B
│   │   │   └── test.png
│   │   └── Individuo_C
│   │       ├── test1.png     
│   └── suspeitos
│       └── criminoso.png
├── debug_output
│   ├── eigenface_0.png
│   ├── eigenface_1.png
│   └── media_face.png
├── pom.xml
├── src
│   └── main
│       └── java
│           ├── Data
│           │   ├── RecognitionResult.java
│           │   └── TrainingData.java
│           ├── FaceRecognizer
│           │   └── FaceRecognizer.java
│           ├── FisherfacesModel
│           │   └── FisherfacesModel.java
│           ├── ImageProcessor
│           │   └── ImageProcessor.java
│           ├── Main.java
│           └── Services
│               ├── DatabaseLoader.java
│               └── VerificationService.java

```

## 🚀 Instalação e Compilação no Linux
1. Instale Git, Java 21 e Maven

```bash
   
sudo apt update
sudo apt install maven
sudo apt install git
```

Verifique versões:
  ```bash
  
  java --version
  mvn --version
  ```

## Clone o projeto

```bash
  cd ~
  git clone https://github.com/JonasSales/projetoAlgebraLinear.git
```

## 🤖 Compilar e Executar (Modo Simplificado)

Agora o projeto possui um script build.sh que faz tudo automaticamente:

✔ mvn clean package silencioso
✔ Compilação
✔ Execução com --enable-preview
✔ Sem precisar configurar classpath

Basta rodar:

```bash

./build.sh
```
Se falar que não há permissão
```bash

chmod +x build.sh
./build.sh
```

## 💡 Execução no Windows

Para rodar:

Dê 2 clicks sobre o arquivo
```
build.bat
```

## 🧪 Executando pelo IntelliJ IDEA (opcional)

Abra o projeto no IntelliJ.

Vá em:
File → Project Structure → Project

Defina:

SDK: Java 21

Em Modules → Language Level, selecione:

21 (Preview)

Execute Main.java.

## 📜 Licença

Uso acadêmico e educacional livre.
