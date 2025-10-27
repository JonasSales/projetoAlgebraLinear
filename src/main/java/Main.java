import Data.RecognitionResult;
import Data.TrainingData;
import FaceRecognizer.FaceRecognizer;
import FisherfacesModel.FisherfacesModel;
import ImageProcessor.ImageProcessor;
import Services.DatabaseLoader;
import Services.VerificationService;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

private static final String DATA_FOLDER = "data";
private static final String DATABASE_FOLDER = "database_criminosos";
private static final String SUSPECTS_FOLDER = "suspeitos";

void main() {
    // --- 1. INICIALIZAÇÃO ---
    ImageProcessor processor = new ImageProcessor();
    DatabaseLoader loader = new DatabaseLoader(processor);
    FisherfacesModel model = new FisherfacesModel();

    String projectDir = System.getProperty("user.dir");
    Path databasePath = Paths.get(projectDir, DATA_FOLDER, DATABASE_FOLDER);
    Path suspectsPath = Paths.get(projectDir, DATA_FOLDER, SUSPECTS_FOLDER);

    try {
        // --- 2. CARREGAMENTO ---
        System.out.printf("Carregando banco de dados de indivíduos de: %s%n", databasePath);
        TrainingData trainingData = loader.loadFromDirectory(databasePath);
        System.out.printf("Banco de dados carregado: %d imagens de referência.%n", trainingData.size());

        if (trainingData.isEmpty()) {
            System.err.println("Nenhuma imagem de treinamento encontrada. Encerrando.");
            return;
        }

        // --- 3. TREINAMENTO ---
        System.out.println("Processando banco de dados (Treinamento Fisherfaces PCA+LDA)...");
        model.train(trainingData);

        if (model.getEigenfaces() != null) {
            System.out.printf("Processamento concluído. Número de Fisherfaces geradas: %d%n", model.getEigenfaces().getColumnDimension());
        } else {
            System.out.println("Processamento falhou ou não gerou Fisherfaces.");
            return;
        }

        // --- 4. VERIFICAÇÃO ---
        FaceRecognizer recognizer = new FaceRecognizer(model, processor);
        VerificationService verificationService = new VerificationService(processor, recognizer);

        System.out.printf("\n--- Iniciando Verificação de Suspeitos na pasta: %s ---%n", suspectsPath);
        List<RecognitionResult> results = verificationService.verifySuspects(suspectsPath);

        if (results.isEmpty()) {
            System.out.println("Nenhum suspeito encontrado para verificação.");
        } else {
            results.forEach(System.out::println);
        }

    } catch (IOException e) {
        System.err.printf("Erro de I/O: %s%n", e.getMessage());
    } catch (Exception e) {
        System.err.printf("Erro inesperado no processamento: %s%n", e.getMessage());
        e.printStackTrace();
    }
}