package FaceRecognizer;

import Data.RecognitionResult;
import FisherfacesModel.FisherfacesModel;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.List;

public class FaceRecognizer {

    private final FisherfacesModel model;
    // O limiar (threshold) é agora uma propriedade da instância
    private final double recognitionThreshold;

    public FaceRecognizer(FisherfacesModel model, ImageProcessor.ImageProcessor processor) {
        this.model = model;
        // A escala das distâncias no espaço LDA (C-1) é
        // completamente diferente da do espaço PCA (K).
        // Este valor (12.0e6) deve ser ajustado/calibrado.
        this.recognitionThreshold = 12.0e6;
    }

    /**
     * Implementa a Fase de Reconhecimento: Projeção e Classificação.
     * Retorna um objeto RecognitionResult com os detalhes.
     * @param inputVector O vetor 1D da imagem de teste.
     * @param fileName O nome do arquivo original para o relatório.
     * @return Um objeto RecognitionResult.
     */
    public RecognitionResult recognize(double[] inputVector, String fileName) {
        if (model.getEigenfaces() == null || model.getMeanVector() == null || model.getProjectedFaces() == null) {
            return new RecognitionResult(fileName, "Modelo não treinado", -1, false);
        }

        // Projeta o vetor de entrada no subespaço Fisher
        double[] coeffs = project(inputVector);

        // Encontra a correspondência mais próxima
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

        // Compara com o limiar
        String label = model.getLabels().get(bestIdx);
        boolean isMatch = bestDist < recognitionThreshold;

        // Se não for uma correspondência, retorne a etiqueta mais próxima, mas marque como "não encontrado"
        String recognizedLabel = isMatch ? label : "Desconhecido";

        // Retorna o resultado estruturado
        return new RecognitionResult(fileName, recognizedLabel, bestDist, isMatch);
    }

    /**
     * Projeta um vetor de imagem no subespaço Fisher.
     */
    private double[] project(double[] inputVector) {
        RealMatrix eigenfaces = model.getEigenfaces();
        double[] mean = model.getMeanVector();

        // diff = inputVector - mean
        RealVector diff = new ArrayRealVector(inputVector).subtract(new ArrayRealVector(mean));

        // projeta no subespaço final: coeffs = eigenfaces^T * diff
        return eigenfaces.transpose().operate(diff).toArray();
    }

    /**
     * Calcula a distância Euclidiana quadrada (mais rápida que sqrt).
     */
    private double euclideanDistanceSquared(double[] v1, double[] v2) {
        double sumOfSquares = 0.0;
        for (int i = 0; i < v1.length; i++) {
            double diffc = v1[i] - v2[i];
            sumOfSquares += diffc * diffc;
        }
        return sumOfSquares;
    }
}