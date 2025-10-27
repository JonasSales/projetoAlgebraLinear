import EigenfacesModel.EigenfacesModel;
import FaceRecognizer.FaceRecognizer;
import ImageProcessor.ImageProcessor;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        ImageProcessor processor = new ImageProcessor();

        List<double[]> trainingVectors = new ArrayList<>();
        List<String> trainingLabels = new ArrayList<>();

        try {
            String projectDir = System.getProperty("user.dir");

            Path p1 = Paths.get(projectDir, "src", "main", "java", "imagensDeTest", "test1.png");
            Path p2 = Paths.get(projectDir, "src", "main", "java", "imagensDeTest", "test2.png");
            Path p3 = Paths.get(projectDir, "src", "main", "java", "imagensDeTest", "test3.png");

            trainingVectors.add(processor.processImage(p1.toFile()));
            trainingLabels.add("Pessoa A");

            trainingVectors.add(processor.processImage(p2.toFile()));
            trainingLabels.add("Pessoa A");

            trainingVectors.add(processor.processImage(p3.toFile()));
            trainingLabels.add("Pessoa B");

            System.out.println("Dados de treinamento carregados: " + trainingVectors.size() + " imagens.");

        } catch (IOException e) {
            System.err.println("Erro ao carregar imagens: " + e.getMessage());
            return;
        }

        int kComponents = 2;
        EigenfacesModel model = new EigenfacesModel(kComponents);

        System.out.println("Iniciando treinamento com K=" + kComponents + "...");
        model.train(trainingVectors, trainingLabels);

        if (model.getEigenfaces() != null) {
            System.out.println("Treinamento concluído. Número de Eigenfaces geradas: " + model.getEigenfaces().getRowDimension());
        } else {
            System.out.println("Treinamento concluído com eigenfaces nulas.");
        }

        FaceRecognizer recognizer = new FaceRecognizer(model, processor);

        try {
            String projectDir = System.getProperty("user.dir");
            Path testPath = Paths.get(projectDir, "src", "main", "java", "imagensDeTest", "test1.png");
            double[] testVectorA = processor.processImage(testPath.toFile());
            String resultA = recognizer.recognize(testVectorA);
            System.out.println(STR."Teste 1 (Pessoa A): \{resultA}");

            Path unknownPath = Paths.get(projectDir, "src", "main", "java", "imagensDeTest", "test2.png");
            double[] testVectorUnknown = processor.processImage(unknownPath.toFile());
            String resultUnknown = recognizer.recognize(testVectorUnknown);
            System.out.println("Teste 2 (Desconhecido): " + resultUnknown);

        } catch (IOException e) {
            System.err.println("Erro ao carregar imagem de teste: " + e.getMessage());
        }
    }
}