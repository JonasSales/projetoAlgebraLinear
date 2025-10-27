import EigenfacesModel.EigenfacesModel;
import FaceRecognizer.FaceRecognizer;
import ImageProcessor.ImageProcessor;


import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Main {

    // --- 1. DEFINIÇÃO DAS PASTAS (Atualizado para o novo contexto) ---
    private static final String DATA_FOLDER = "data";
    // Pasta do banco de dados de indivíduos registrados
    private static final String DATABASE_FOLDER = "database_criminosos";
    // Pasta das imagens a serem verificadas
    private static final String SUSPECTS_FOLDER = "suspeitos";

    public static void main(String[] args) {
        ImageProcessor processor = new ImageProcessor();
        List<double[]> trainingVectors = new ArrayList<>();
        List<String> trainingLabels = new ArrayList<>();

        // Obtém o caminho raiz do projeto
        String projectDir = System.getProperty("user.dir");
        Path databasePath = Paths.get(projectDir, DATA_FOLDER, DATABASE_FOLDER); // Caminho atualizado
        Path suspectsPath = Paths.get(projectDir, DATA_FOLDER, SUSPECTS_FOLDER); // Caminho atualizado

        System.out.println(STR."Carregando banco de dados de indivíduos de: \{databasePath}");

        // --- 2. CARREGAMENTO DO BANCO DE DADOS ---
        try {
            // A função é a mesma, mas o contexto muda: "treinar" = "carregar BD"
            loadTrainingData(databasePath, processor, trainingVectors, trainingLabels);
            System.out.println(STR."Banco de dados carregado: \{trainingVectors.size()} imagens de referência.");

            if (trainingVectors.isEmpty()) {
                System.err.println("Nenhuma imagem encontrada no banco de dados.");
                System.err.println(STR."Verifique se a pasta '\{databasePath}' existe e contém subpastas com imagens.");
                return;
            }

        } catch (IOException e) {
            System.err.println(STR."Erro ao carregar banco de dados: \{e.getMessage()}");
            return;
        }

        // --- 3. PROCESSAMENTO (Treinamento do Modelo PCA) ---
        int kComponents = Math.min(Math.max(1, trainingVectors.size() - 1), 50);
        EigenfacesModel model = new EigenfacesModel(kComponents); //

        System.out.println(STR."Processando banco de dados (Treinamento com K=\{kComponents})...");
        model.train(trainingVectors, trainingLabels); //

        if (model.getEigenfaces() != null) {
            System.out.println(STR."Processamento concluído. Número de Eigenfaces geradas: \{model.getEigenfaces().getColumnDimension()}");
        } else {
            System.out.println("Processamento falhou ou não gerou Eigenfaces.");
            return;
        }

        // --- 4. VERIFICAÇÃO DE SUSPEITOS ---
        FaceRecognizer recognizer = new FaceRecognizer(model, processor); //
        System.out.println(STR."\n--- Iniciando Verificação de Suspeitos na pasta: \{suspectsPath} ---");

        try {
            // O nome da função mudou para clareza
            runSuspectVerification(suspectsPath, processor, recognizer);
        } catch (IOException e) {
            System.err.println(STR."Erro ao verificar imagens de suspeitos: \{e.getMessage()}");
        }
    }

    /**
     * Carrega as imagens de referência do banco de dados.
     * A estrutura esperada é: databaseDir -> [Nome_Individuo] -> [image.png]
     * Ex: data/database_criminosos/Individuo_X/rosto1.png
     */
    public static void loadTrainingData(Path trainDir, ImageProcessor processor, List<double[]> outVectors, List<String> outLabels) throws IOException {

        if (!Files.exists(trainDir) || !Files.isDirectory(trainDir)) {
            throw new IOException(STR."Diretório do banco de dados não encontrado: \{trainDir}");
        }

        // Itera sobre as subpastas (ex: "Individuo_X", "Individuo_Y")
        try (DirectoryStream<Path> labelDirs = Files.newDirectoryStream(trainDir, Files::isDirectory)) {
            for (Path labelDir : labelDirs) {
                String label = labelDir.getFileName().toString();
                System.out.println(STR."  Registrando indivíduo: \{label}");

                // Itera sobre os arquivos de imagem dentro da pasta do indivíduo
                try (DirectoryStream<Path> imageFiles = Files.newDirectoryStream(labelDir)) {
                    for (Path imageFile : imageFiles) {
                        String fileName = imageFile.getFileName().toString().toLowerCase();

                        if (fileName.endsWith(".png") || fileName.endsWith(".jpg") || fileName.endsWith(".jpeg")) {
                            try {
                                // Processa a imagem e adiciona ao "treinamento"
                                outVectors.add(processor.processImage(imageFile.toFile())); //
                                outLabels.add(label);
                            } catch (IOException e) {
                                System.err.println(STR."  [Aviso] Falha ao processar imagem \{imageFile.getFileName()}: \{e.getMessage()}");
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Executa a verificação em todos os arquivos da pasta de suspeitos.
     * Esta é a "Fase de Reconhecimento".
     */
    public static void runSuspectVerification(Path testDir, ImageProcessor processor, FaceRecognizer recognizer) throws IOException {

        if (!Files.exists(testDir) || !Files.isDirectory(testDir)) {
            System.err.println(STR."Diretório de suspeitos não encontrado: \{testDir}");
            return;
        }

        try (DirectoryStream<Path> testFiles = Files.newDirectoryStream(testDir)) {
            for (Path testFile : testFiles) {
                String fileName = testFile.getFileName().toString().toLowerCase();

                if (fileName.endsWith(".png") || fileName.endsWith(".jpg") || fileName.endsWith(".jpeg")) {
                    try {
                        double[] testVector = processor.processImage(testFile.toFile());
                        String result = recognizer.recognize(testVector);
                        if (result.startsWith("Desconhecido")) {
                            System.out.println(STR."  VERIFICANDO [\{testFile.getFileName()}]: Status: NÃO ENCONTRADO. \{result}");
                        } else {
                            System.out.println(STR."  ALERTA! VERIFICANDO [\{testFile.getFileName()}]: POSSÍVEL CORRESPONDÊNCIA ENCONTRADA! -> \{result}");
                        }

                    } catch (IOException e) {
                        System.err.println(STR."  [Aviso] Falha ao verificar imagem \{testFile.getFileName()}: \{e.getMessage()}");
                    }
                }
            }
        }
    }
}