import Data.RecognitionResult;
import Data.TrainingData;
import FaceRecognizer.FaceRecognizer;
import FisherfacesModel.FisherfacesModel;
import ImageProcessor.ImageProcessor;
import Services.DatabaseLoader;
import Services.VerificationService;
import org.apache.commons.math3.linear.RealMatrix;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

// Padrões apenas se nenhum argumento for passado
// private static final String DEFAULT_DB = "data/database_criminosos";
// private static final String DEFAULT_SUSPECTS = "data/suspeitos";

void main(String[] args) {
    // Configuração de caminhos (Ajuste conforme seu projeto)
    String dbPathStr = "data/database_criminosos";
    String suspectsPathStr = "data/suspeitos";
    double threshold = 12.0e6;

    if (args.length >= 1) dbPathStr = args[0];
    if (args.length >= 2) suspectsPathStr = args[1];

    Path databasePath = Paths.get(System.getProperty("user.dir"), dbPathStr);
    Path suspectsPath = Paths.get(System.getProperty("user.dir"), suspectsPathStr);

    System.out.println("=== SISTEMA DE RECONHECIMENTO FACIAL (DEMONSTRAÇÃO ACADÊMICA) ===");

    ImageProcessor processor = new ImageProcessor();
    DatabaseLoader loader = new DatabaseLoader(processor);
    FisherfacesModel model = new FisherfacesModel();

    try {
        System.out.println("\n--- 1. Carregamento de Imagens ---");
        TrainingData trainingData = loader.loadFromDirectory(databasePath);

        if (trainingData.isEmpty()) {
            System.err.println("[ERRO] Nenhuma imagem encontrada. Verifique o caminho.");
            return;
        }

        System.out.println("\n--- 2. Treinamento (Álgebra Linear) ---");
        model.train(trainingData);

        // --- DIDÁTICO: SALVAR EIGENFACES ---
        System.out.println("\n--- 3. Visualização Matemática ---");
        System.out.println("Gerando imagens das Eigenfaces (Rostos Característicos)...");
        RealMatrix W = model.getEigenfaces();
        // Salvar a face média
        processor.saveVectorAsImage(model.getMeanVector(), "debug_output/media_face.png");

        // Salvar as primeiras 5 eigenfaces (componentes principais)
        for (int i = 0; i < Math.min(5, W.getColumnDimension()); i++) {
            double[] eigenVector = W.getColumn(i);
            processor.saveVectorAsImage(eigenVector, "debug_output/eigenface_" + i + ".png");
        }
        System.out.println("Verifique a pasta 'debug_output' para ver como o algoritmo 'vê' os rostos.");

        // 4. Reconhecimento
        System.out.println("\n--- 4. Reconhecimento e Testes ---");
        FaceRecognizer recognizer = new FaceRecognizer(model, processor, threshold);
        VerificationService verificationService = new VerificationService(processor, recognizer);

        verificationService.verifySuspects(suspectsPath);

    } catch (Exception e) {
        System.err.println("Erro Geral: " + e.getMessage());
        e.printStackTrace();
    }
}