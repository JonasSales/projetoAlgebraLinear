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

// Padrões apenas se nenhum argumento for passado
private static final String DEFAULT_DB = "data/database_criminosos";
private static final String DEFAULT_SUSPECTS = "data/suspeitos";

void main(String[] args) {
    // Configuração de caminhos via argumentos (CLI) ou fallback
    String dbPathStr = (args.length >= 1) ? args[0] : DEFAULT_DB;
    String suspectsPathStr = (args.length >= 2) ? args[1] : DEFAULT_SUSPECTS;
    // Permite ajuste do threshold via argumento 3
    double threshold = (args.length >= 3) ? Double.parseDouble(args[2]) : 15.0e6;

    Path databasePath = Paths.get(System.getProperty("user.dir"), dbPathStr);
    Path suspectsPath = Paths.get(System.getProperty("user.dir"), suspectsPathStr);

    System.out.println("=== SISTEMA DE RECONHECIMENTO FACIAL (FISHERFACES) ===");
    System.out.println("Configuração:");
    System.out.println("  - Banco de Treino: " + databasePath);
    System.out.println("  - Pasta Suspeitos: " + suspectsPath);
    System.out.println("  - Limiar Distância: " + threshold);

    // 1. Inicialização
    ImageProcessor processor = new ImageProcessor();
    DatabaseLoader loader = new DatabaseLoader(processor);
    FisherfacesModel model = new FisherfacesModel();

    try {
        // 2. Carregamento (Paralelo)
        long startLoad = System.currentTimeMillis();
        TrainingData trainingData = loader.loadFromDirectory(databasePath);
        long endLoad = System.currentTimeMillis();

        if (trainingData.isEmpty()) {
            System.err.println("[ERRO] Nenhuma imagem encontrada. Verifique o caminho.");
            return;
        }
        System.out.printf("Carregamento concluído em %d ms. Imagens: %d%n", (endLoad - startLoad), trainingData.size());

        // 3. Treinamento
        System.out.println("Iniciando treinamento do modelo...");
        model.train(trainingData);
        System.out.printf("Treinamento OK. Fisherfaces: %d%n", model.getEigenfaces().getColumnDimension());

        // 4. Reconhecimento
        FaceRecognizer recognizer = new FaceRecognizer(model, processor, threshold);
        VerificationService verificationService = new VerificationService(processor, recognizer);

        System.out.println("\n--- Resultados da Verificação ---");
        List<RecognitionResult> results = verificationService.verifySuspects(suspectsPath);

        if (results.isEmpty()) {
            System.out.println("Nenhum arquivo processado na pasta de suspeitos.");
        } else {
            for (RecognitionResult r : results) {
                System.out.println(r);
            }
        }

    } catch (IOException e) {
        System.err.println("Erro Crítico de I/O: " + e.getMessage());
    } catch (Exception e) {
        System.err.println("Erro Geral: " + e.getMessage());
        e.printStackTrace();
    }
}