package Services;

import Data.RecognitionResult;
import FaceRecognizer.FaceRecognizer;
import ImageProcessor.ImageProcessor;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Encapsula a lógica de verificação de imagens de suspeitos
 * contra o modelo treinado.
 */
public record VerificationService(ImageProcessor processor, FaceRecognizer recognizer) {

    /**
     * Executa a verificação em todos os arquivos da pasta de suspeitos.
     */
    public List<RecognitionResult> verifySuspects(Path testDir) throws IOException {
        if (!Files.exists(testDir) || !Files.isDirectory(testDir)) {
            System.err.printf("Diretório de suspeitos não encontrado: %s%n", testDir);
            return new ArrayList<>();
        }

        List<RecognitionResult> results = new ArrayList<>();

        try (DirectoryStream<Path> testFiles = Files.newDirectoryStream(testDir)) {
            for (Path testFile : testFiles) {
                if (isImageFile(testFile)) {
                    try {
                        double[] testVector = processor.processImage(testFile.toFile());
                        // A lógica de reconhecimento foi movida para o FaceRecognizer
                        RecognitionResult result = recognizer.recognize(testVector, testFile.getFileName().toString());
                        results.add(result);

                    } catch (IOException e) {
                        System.err.printf("  [Aviso] Falha ao verificar imagem %s: %s%n", testFile.getFileName(), e.getMessage());
                    }
                }
            }
        }
        return results;
    }

    private boolean isImageFile(Path file) {
        String fileName = file.getFileName().toString().toLowerCase();
        return fileName.endsWith(".png") || fileName.endsWith(".jpg") || fileName.endsWith(".jpeg");
    }
}
