import FaceRecognizer.FaceRecognizer;
import FisherfacesModel.FisherfacesModel;
import ImageProcessor.ImageProcessor;


import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Main {
    private static final String DATA_FOLDER = "data";
    private static final String DATABASE_FOLDER = "database_criminosos";
    private static final String SUSPECTS_FOLDER = "suspeitos";

    public static void main(String[] args) {
        ImageProcessor processor = new ImageProcessor();
        List<double[]> trainingVectors = new ArrayList<>();
        List<String> trainingLabels = new ArrayList<>();

        String projectDir = System.getProperty("user.dir");
        Path databasePath = Paths.get(projectDir, DATA_FOLDER, DATABASE_FOLDER);
        Path suspectsPath = Paths.get(projectDir, DATA_FOLDER, SUSPECTS_FOLDER);

        System.out.println(STR."Carregando banco de dados de indivíduos de: \{databasePath}");

        try {
            loadTrainingData(databasePath, processor, trainingVectors, trainingLabels);
            System.out.println(STR."Banco de dados carregado: \{trainingVectors.size()} imagens de referência.");

            if (trainingVectors.isEmpty()) {
                return;
            }
        } catch (IOException e) {
            System.err.println(STR."Erro ao carregar banco de dados: \{e.getMessage()}");
            return;
        }

        // --- 3. PROCESSAMENTO (Treinamento do Modelo Fisherfaces) ---

        // REMOVER o cálculo de kComponents. O modelo determina-o sozinho.
        // int kComponents = Math.min(Math.max(1, trainingVectors.size() - 1), 50);

        // MUDAR a instanciação do modelo
        FisherfacesModel model = new FisherfacesModel(); // Antes: new EigenfacesModel(kComponents)

        // MUDAR a mensagem de log
        System.out.println("Processando banco de dados (Treinamento Fisherfaces PCA+LDA)...");
        model.train(trainingVectors, trainingLabels); //

        if (model.getEigenfaces() != null) {
            // Esta linha agora imprime o N. de componentes LDA (C-1)
            System.out.println(STR."Processamento concluído. Número de Fisherfaces geradas: \{model.getEigenfaces().getColumnDimension()}");
        } else {
            System.out.println("Processamento falhou ou não gerou Fisherfaces.");
            return;
        }

        // --- 4. VERIFICAÇÃO DE SUSPEITOS ---
        // Esta linha funciona, pois o FaceRecognizer foi atualizado
        FaceRecognizer recognizer = new FaceRecognizer(model, processor);
        System.out.println(STR."\n--- Iniciando Verificação de Suspeitos na pasta: \{suspectsPath} ---");

        try {
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