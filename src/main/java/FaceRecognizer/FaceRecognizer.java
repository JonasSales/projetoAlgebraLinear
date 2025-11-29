package FaceRecognizer;

import Data.RecognitionResult;
import FisherfacesModel.FisherfacesModel;
import ImageProcessor.ImageProcessor;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.List;

public class FaceRecognizer {

    private final FisherfacesModel model;
    private double recognitionThreshold;

    // Classe auxiliar interna para ranking
    private record MatchCandidate(String label, double distance) {}

    public FaceRecognizer(FisherfacesModel model, ImageProcessor processor, double threshold) {
        this.model = model;
        this.recognitionThreshold = threshold;
    }

    public FaceRecognizer(FisherfacesModel model, ImageProcessor processor) {
        this(model, processor, 12.0e6);
    }

    public void setRecognitionThreshold(double value) {
        this.recognitionThreshold = value;
    }

    public RecognitionResult recognize(double[] inputVector, String fileName) {
        if (model.getEigenfaces() == null || model.getMeanVector() == null) {
            return new RecognitionResult(fileName, "Modelo não treinado", -1, false);
        }

        double[] coeffs = project(inputVector);

        // Calcular todas as distâncias para criar um ranking
        List<MatchCandidate> ranking = new ArrayList<>();
        List<double[]> projections = model.getProjectedFaces();
        List<String> labels = model.getLabels();

        for (int i = 0; i < projections.size(); i++) {
            double d = euclideanDistanceSquared(coeffs, projections.get(i));
            ranking.add(new MatchCandidate(labels.get(i), d));
        }

        // Ordenar por menor distância
        ranking.sort((c1, c2) -> Double.compare(c1.distance, c2.distance));

        // --- EXIBIÇÃO DIDÁTICA DO RANKING ---
        System.out.println("--------------------------------------------------");
        System.out.printf("Analisando imagem: %s%n", fileName);
        System.out.println("Ranking de Proximidade (Cálculo de Distância Euclidiana):");
        for (int i = 0; i < Math.min(3, ranking.size()); i++) {
            MatchCandidate c = ranking.get(i);
            System.out.printf("  %dº. Candidato: %-15s | Distância: %.2f%n", (i+1), c.label, c.distance);
        }

        MatchCandidate best = ranking.getFirst();
        boolean isMatch = best.distance < recognitionThreshold;

        if (isMatch) {
            System.out.println("  -> CONCLUSÃO: Correspondência Confirmada!");
        } else {
            System.out.println("  -> CONCLUSÃO: Distância muito alta. Desconhecido.");
        }
        System.out.println("--------------------------------------------------");

        return new RecognitionResult(fileName, isMatch ? best.label : "Desconhecido", best.distance, isMatch);
    }

    private double[] project(double[] inputVector) {
        RealMatrix eigenfaces = model.getEigenfaces();
        double[] mean = model.getMeanVector();
        RealVector diff = new ArrayRealVector(inputVector).subtract(new ArrayRealVector(mean));
        return eigenfaces.transpose().operate(diff).toArray();
    }

    private double euclideanDistanceSquared(double[] v1, double[] v2) {
        double sum = 0.0;
        for (int i = 0; i < v1.length; i++) {
            double d = v1[i] - v2[i];
            sum += d * d;
        }
        return sum;
    }
}