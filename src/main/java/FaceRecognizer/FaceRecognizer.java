package FaceRecognizer;

import FisherfacesModel.FisherfacesModel; // Antes: EigenfacesModel
import ImageProcessor.ImageProcessor;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.List;

public class FaceRecognizer {

    // Mudar o tipo do modelo
    private final FisherfacesModel model; // Antes: EigenfacesModel

    // Mudar o construtor
    public FaceRecognizer(FisherfacesModel model, ImageProcessor processor) { // Antes: EigenfacesModel
        this.model = model;
    }

    /**
     * Implementa a Fase de Reconhecimento: Projeção e Classificação.
     * Retorna o nome da pessoa reconhecida junto com a distância euclidiana (quadrada).
     * @param inputVector O vetor 1D da imagem de teste.
     * @return Uma string contendo o rótulo e o valor mensurável (distância).
     */
    public String recognize(double[] inputVector) {
        // ESTE CÓDIGO FUNCIONA EXATAMENTE COMO ANTES!
        // model.getEigenfaces() agora retorna W_final (W_pca * W_lda)
        // model.getMeanVector() retorna a média global
        // model.getProjectedFaces() retorna os vetores de treino no espaço LDA (C-1)

        RealMatrix eigenfaces = model.getEigenfaces();
        double[] mean = model.getMeanVector();

        if (eigenfaces == null || mean == null || model.getProjectedFaces() == null) return "Modelo não treinado";

        // calcula diff (dim)
        RealVector diff = new ArrayRealVector(inputVector).subtract(new ArrayRealVector(mean));

        // projeta no subespaço final: coeffs = eigenfaces^T * diff (tamanho C-1)
        RealVector coeffs = eigenfaces.transpose().operate(diff);
        double[] c = coeffs.toArray();

        // compara com projeções treinadas (todas têm tamanho C-1)
        double bestDist = Double.POSITIVE_INFINITY;
        int bestIdx = -1;
        List<double[]> projections = model.getProjectedFaces();
        for (int i = 0; i < projections.size(); i++) {
            double[] p = projections.get(i);
            if (p == null || p.length != c.length) continue;
            double d = 0;
            for (int j = 0; j < c.length; j++) {
                double diffc = c[j] - p[j];
                d += diffc * diffc;
            }
            if (d < bestDist) {
                bestDist = d;
                bestIdx = i;
            }
        }

        String label = "Desconhecido";
        // NOTA: O limiar (threshold) provavelmente precisará ser ajustado!
        // A escala das distâncias no espaço LDA (C-1 dimensões) é
        // completamente diferente da do espaço PCA (K dimensões).
        // Sugiro começar com um valor muito mais baixo ou testar.
        double threshold = 12.0e6; // Valor anterior: 15e6. Ajuste conforme necessário.

        if (bestIdx != -1) {
            label = bestDist < threshold ? model.getLabels().get(bestIdx) : "Desconhecido";
        }

        return String.format("%s (Distância Euclidiana Quadrada: %.2f)", label, bestDist);
    }

    // ... método euclideanDistance (sem alterações) ...
    private double euclideanDistance(double[] v1, double[] v2) {
        double sumOfSquares = 0.0;
        for (int i = 0; i < v1.length; i++) {
            sumOfSquares += Math.pow(v1[i] - v2[i], 2);
        }
        return Math.sqrt(sumOfSquares);
    }
}