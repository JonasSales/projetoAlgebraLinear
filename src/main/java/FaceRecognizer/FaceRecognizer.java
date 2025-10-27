package FaceRecognizer;

import EigenfacesModel.EigenfacesModel;
import ImageProcessor.ImageProcessor;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.util.List;

public class FaceRecognizer {

    private final EigenfacesModel model;
    private final ImageProcessor imageProcessor;

    public FaceRecognizer(EigenfacesModel model, ImageProcessor processor) {
        this.model = model;
        this.imageProcessor = processor;
    }

    /**
     * Implementa a Fase de Reconhecimento: Projeção e Classificação.
     * @param testVector O vetor 1D da imagem de teste.
     * @return O nome da pessoa reconhecida ou "Desconhecido".
     */
    public String recognize(double[] testVector) {
        if (model.getEigenfaces() == null || model.getMeanFace() == null) {
            return "Modelo não treinado.";
        }
        
        // 1. Centralização (Passos 1 a 4 da Fase de Treinamento)
        RealVector newVector = new ArrayRealVector(testVector);
        RealVector centeredVector = newVector.subtract(model.getMeanFace()); // Vetor Diferença

        // 2. Projeção da Imagem de Teste (no Espaço Facial)
        // Projeção = Eigenfaces * Vetor_Diferença_Transposto
        RealVector testProjection = model.getEigenfaces().operate(centeredVector);
        double[] testCoefficients = testProjection.toArray();

        // 3. Classificação (Comparação): Distância Euclidiana
        double minDistance = Double.MAX_VALUE;
        String recognizedLabel = "Desconhecido";
        
        List<double[]> projectedFaces = model.getProjectedFaces();
        List<String> labels = model.getLabels();

        for (int i = 0; i < projectedFaces.size(); i++) {
            double[] trainingCoefficients = projectedFaces.get(i);
            String label = labels.get(i);
            
            // Cálculo da Distância Euclidiana entre os vetores de coeficientes
            double distance = euclideanDistance(testCoefficients, trainingCoefficients);

            if (distance < minDistance) {
                minDistance = distance;
                recognizedLabel = label;
            }
        }
        
        // 4. Decisão: A menor distância define a pessoa.
        // *OPCIONAL*: Adicionar um limiar (threshold) para classificar como "Desconhecido"
        // se minDistance for muito grande.
        double threshold = 50.0; // Valor a ser ajustado empiricamente
        if (minDistance > threshold) {
             return "Desconhecido (Distância: " + String.format("%.2f", minDistance) + ")";
        }
        
        return recognizedLabel + " (Distância: " + String.format("%.2f", minDistance) + ")";
    }

    /**
     * Calcula a Distância Euclidiana entre dois vetores.
     */
    private double euclideanDistance(double[] v1, double[] v2) {
        double sumOfSquares = 0.0;
        for (int i = 0; i < v1.length; i++) {
            sumOfSquares += Math.pow(v1[i] - v2[i], 2);
        }
        return Math.sqrt(sumOfSquares);
    }
}
