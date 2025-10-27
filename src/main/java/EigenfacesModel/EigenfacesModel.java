package EigenfacesModel;

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.correlation.Covariance;

import java.util.List;
import java.util.ArrayList;
import java.util.Comparator;

public class EigenfacesModel {

    private RealMatrix eigenfaces; // As K Eigenfaces selecionadas
    private RealVector meanFace;   // A Face Média (Ψ)
    private final List<double[]> projectedFaces; // Projeções (Coeficientes) das faces de treinamento
    private final List<String> labels; // Rótulos/Nomes associados às faces (Identidade)
    private final int K_COMPONENTS; // Número de componentes principais (Eigenfaces) a reter

    public EigenfacesModel(int k) {
        this.K_COMPONENTS = k;
        this.projectedFaces = new ArrayList<>();
        this.labels = new ArrayList<>();
    }

    /**
     * Implementa a Fase de Treinamento: PCA para Eigenfaces.
     * Passos 3 a 8 do documento.
     * @param trainingVectors Lista de vetores de imagens de treinamento.
     * @param faceLabels Rótulos/Nomes correspondentes.
     */
    public void train(List<double[]> trainingVectors, List<String> labels) {
        int m = trainingVectors.size();
        if (m == 0) throw new IllegalArgumentException("Nenhuma imagem de treinamento.");
        int dim = trainingVectors.get(0).length;

        // calcula média
        double[] mean = new double[dim];
        for (double[] v : trainingVectors) {
            if (v.length != dim) throw new IllegalArgumentException("Dimensões inconsistentes das imagens.");
            for (int i = 0; i < dim; i++) mean[i] += v[i];
        }
        for (int i = 0; i < dim; i++) mean[i] /= m;
        this.meanFace = new ArrayRealVector(mean);

        // monta A (dim x m) com vetores centrados
        RealMatrix A = new Array2DRowRealMatrix(dim, m);
        for (int j = 0; j < m; j++) {
            double[] diff = new double[dim];
            double[] v = trainingVectors.get(j);
            for (int i = 0; i < dim; i++) diff[i] = v[i] - mean[i];
            A.setColumn(j, diff);
        }

        // calcula matriz pequena C = A^T * A (m x m)
        RealMatrix C = A.transpose().multiply(A);

        // eigen decomposition de C
        EigenDecomposition ed = new EigenDecomposition(C);
        double[] ev = ed.getRealEigenvalues();

        // coleta pares (valor, vetor) e ordena decrescente
        List<Integer> idx = new ArrayList<>();
        for (int i = 0; i < ev.length; i++) idx.add(i);
        idx.sort((i, j) -> Double.compare(ev[j], ev[i]));

        // ajusta k (não pode ser maior que m)
        int k = Math.min(this.K_COMPONENTS, m);
        List<double[]> eigenfacesList = new ArrayList<>();

        for (int t = 0; t < k; t++) {
            int i = idx.get(t);
            double lambda = ev[i];
            if (lambda <= 1e-12) break; // ignora autovalores muito pequenos
            RealVector vSmall = ed.getEigenvector(i);
            // u = A * vSmall
            RealVector u = A.operate(vSmall);
            double norm = Math.sqrt(u.dotProduct(u));
            if (norm < 1e-12) continue;
            // normaliza u
            double[] uArr = u.mapDivide(norm).toArray();
            eigenfacesList.add(uArr);
        }

        // monta matriz eigenfaces (dim x k')
        int kActual = eigenfacesList.size();
        if (kActual == 0) {
            this.eigenfaces = null;
            return;
        }
        RealMatrix eigenfacesMat = new Array2DRowRealMatrix(dim, kActual);
        for (int c = 0; c < kActual; c++) eigenfacesMat.setColumn(c, eigenfacesList.get(c));
        this.eigenfaces = eigenfacesMat;

        // projeções dos vetores de treino no espaço das eigenfaces
        this.projectedFaces.clear();
        for (int j = 0; j < m; j++) {
            RealVector diff = new ArrayRealVector(A.getColumn(j));
            RealVector coeffs = eigenfacesMat.transpose().operate(diff);
            this.projectedFaces.add(coeffs.toArray());
        }

        // salva labels (assume ordem correspondente)
        this.labels.clear();
        this.labels.addAll(labels);
    }

    /**
     * Calcula o vetor médio das imagens de treinamento.
     */
    private RealVector calculateMeanFace(RealMatrix dataMatrix) {
        int numImages = dataMatrix.getRowDimension();
        int vectorSize = dataMatrix.getColumnDimension();
        double[] meanData = new double[vectorSize];
        for (int j = 0; j < vectorSize; j++) {
            double sum = 0.0;
            for (int i = 0; i < numImages; i++) {
                sum += dataMatrix.getEntry(i, j);
            }
            meanData[j] = sum / numImages;
        }
        return new ArrayRealVector(meanData);
    }

    /**
     * Centraliza os dados subtraindo a Face Média de cada vetor de imagem.
     */
    private RealMatrix centerData(RealMatrix dataMatrix, RealVector mean) {
        RealMatrix centered = dataMatrix.copy();
        for (int i = 0; i < centered.getRowDimension(); i++) {
            centered.setRowVector(i, centered.getRowVector(i).subtract(mean));
        }
        return centered;
    }

    // Classe auxiliar para ordenação de Autovetores e Autovalores
    private static class EigenPair {
        double eigenvalue;
        RealVector eigenvector;

        public EigenPair(double eigenvalue, RealVector eigenvector) {
            this.eigenvalue = eigenvalue;
            this.eigenvector = eigenvector;
        }
    }
    
    // Getters para uso na fase de Reconhecimento
    public RealMatrix getEigenfaces() {
        return eigenfaces;
    }

    public RealVector getMeanFace() {
        return meanFace;
    }

    public List<double[]> getProjectedFaces() {
        return projectedFaces;
    }

    public List<String> getLabels() {
        return labels;
    }
}
