package Services;

import Data.TrainingData;
import ImageProcessor.ImageProcessor;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Encapsula a lógica de carregamento e processamento
 * das imagens do banco de dados de treinamento.
 */
public record DatabaseLoader(ImageProcessor processor) {

    /**
     * Carrega as imagens de referência do banco de dados.
     * A estrutura esperada é: databaseDir -> [Nome_Individuo] -> [image.png]
     */
    public TrainingData loadFromDirectory(Path trainDir) throws IOException {
        if (!Files.exists(trainDir) || !Files.isDirectory(trainDir)) {
            throw new IOException(String.format("Diretório do banco de dados não encontrado: %s", trainDir));
        }

        List<double[]> outVectors = new ArrayList<>();
        List<String> outLabels = new ArrayList<>();

        // Itera sobre as subpastas (ex: "Individuo_X", "Individuo_Y")
        try (DirectoryStream<Path> labelDirs = Files.newDirectoryStream(trainDir, Files::isDirectory)) {
            for (Path labelDir : labelDirs) {
                String label = labelDir.getFileName().toString();
                System.out.printf("  Registrando indivíduo: %s%n", label);

                // Itera sobre os arquivos de imagem dentro da pasta do indivíduo
                try (DirectoryStream<Path> imageFiles = Files.newDirectoryStream(labelDir)) {
                    for (Path imageFile : imageFiles) {
                        if (isImageFile(imageFile)) {
                            try {
                                // Processa a imagem e adiciona ao "treinamento"
                                outVectors.add(processor.processImage(imageFile.toFile()));
                                outLabels.add(label);
                            } catch (IOException e) {
                                System.err.printf("  [Aviso] Falha ao processar imagem %s: %s%n", imageFile.getFileName(), e.getMessage());
                            }
                        }
                    }
                }
            }
        }
        return new TrainingData(outVectors, outLabels);
    }

    private boolean isImageFile(Path file) {
        String fileName = file.getFileName().toString().toLowerCase();
        return fileName.endsWith(".png") || fileName.endsWith(".jpg") || fileName.endsWith(".jpeg");
    }
}
