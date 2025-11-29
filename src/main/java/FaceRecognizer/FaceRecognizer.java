package FaceRecognizer;

import Data.RecognitionResult;
import FisherfacesModel.FisherfacesModel;
import ImageProcessor.ImageProcessor;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.List;

public class FaceRecognizer {

    private final FisherfacesModel model;
    private double recognitionThreshold;

    public FaceRecognizer(FisherfacesModel model, ImageProcessor processor, double threshold) {
        this.model = model;
        this.recognitionThreshold = threshold;
    }

    // Sobrecarga para valor padrão
    public FaceRecognizer(FisherfacesModel model, ImageProcessor processor) {
        this(model, processor, 12.0e6); // Valor padrão original, ajuste conforme necessário
    }

    public void setRecognitionThreshold(double value) {
        this.recognitionThreshold = value;
    }

    public RecognitionResult recognize(double[] inputVector, String fileName) {
        if (model.getEigenfaces() == null || model.getMeanVector() == null) {
            return new RecognitionResult(fileName, "Modelo não treinado", -1, false);
        }

        double[] coeffs = project(inputVector);

        double bestDist = Double.POSITIVE_INFINITY;
        int bestIdx = -1;
        List<double[]> projections = model.getProjectedFaces();

        for (int i = 0; i < projections.size(); i++) {
            double d = euclideanDistanceSquared(coeffs, projections.get(i));
            if (d < bestDist) {
                bestDist = d;
                bestIdx = i;
            }
        }

        if (bestIdx == -1) {
            return new RecognitionResult(fileName, "Desconhecido", bestDist, false);
        }

        String label = model.getLabels().get(bestIdx);
        // Lógica simples: se distância < limiar, é match.
        boolean isMatch = bestDist < recognitionThreshold;

        return new RecognitionResult(fileName, isMatch ? label : "Desconhecido", bestDist, isMatch);
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