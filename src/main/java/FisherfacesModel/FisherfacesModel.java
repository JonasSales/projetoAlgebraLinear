package FisherfacesModel;

import Data.TrainingData;
import org.apache.commons.math3.linear.*;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

public class FisherfacesModel {

    // Campos privados (bom encapsulamento)
    private RealMatrix eigenfaces; // W_final = W_pca * W_lda
    private RealVector meanFace;   // Média Global (Ψ)
    private final List<double[]> projectedFaces;
    private final List<String> labels;

    // Matrizes de projeção intermediárias (agora campos da classe)
    private RealMatrix w_pca;
    private RealMatrix w_lda;

    public FisherfacesModel() {
        this.projectedFaces = new ArrayList<>();
        this.labels = new ArrayList<>();
    }

    /**
     * Método público principal que orquestra o treinamento.
     */
    public void train(TrainingData data) {
        int m = data.size(); // N (Número total de amostras)
        if (m == 0) throw new IllegalArgumentException("Nenhuma imagem de treinamento.");
        int dim = data.vectors().get(0).length;

        // --- Agrupar dados ---
        Map<String, List<double[]>> classes = agruparPorClasse(data.vectors(), data.labels());
        int C = classes.size(); // C (Número de classes/indivíduos)
        if (C <= 1) throw new IllegalArgumentException("O treinamento LDA requer pelo menos 2 classes (indivíduos).");

        // --- PASSO 1: PCA ---
        // k_pca = N - C
        int k_pca = m - C;
        Map<String, List<RealVector>> projectedClasses = executarPCA(data.vectors(), classes, dim, m, k_pca);

        if (this.w_pca == null) {
            throw new RuntimeException(
                    String.format("PCA não gerou componentes. k_pca (N-C) = %d. Você precisa de mais imagens do que indivíduos (N > C).", k_pca)
            );
        }

        int k_pca_actual = this.w_pca.getColumnDimension();

        // --- PASSO 2: LDA ---
        // k_lda = C - 1
        int k_lda = C - 1;
        executarLDA(projectedClasses, k_pca_actual, k_lda, m);

        // --- PASSO 3: Finalização ---
        // W_final = W_pca * W_lda
        this.eigenfaces = this.w_pca.multiply(this.w_lda);

        // Projetar dados de treinamento no espaço final
        projetarDadosFinais(projectedClasses);
    }

    // --- MÉTODOS PRIVADOS DE TREINAMENTO ---

    private Map<String, List<double[]>> agruparPorClasse(List<double[]> trainingVectors, List<String> faceLabels) {
        Map<String, List<double[]>> classes = new HashMap<>();
        for (int i = 0; i < trainingVectors.size(); i++) {
            classes.computeIfAbsent(faceLabels.get(i), k -> new ArrayList<>()).add(trainingVectors.get(i));
        }
        return classes;
    }

    private Map<String, List<RealVector>> executarPCA(List<double[]> trainingVectors, Map<String, List<double[]>> classes, int dim, int m, int k_pca) {
        // 1.1. Calcular média global (meanFace)
        double[] mean = new double[dim];
        for (double[] v : trainingVectors) {
            for (int i = 0; i < dim; i++) mean[i] += v[i];
        }
        for (int i = 0; i < dim; i++) mean[i] /= m;
        this.meanFace = new ArrayRealVector(mean);

        // 1.2. Montar Matriz A (dados centralizados) [dim x m]
        RealMatrix A = new Array2DRowRealMatrix(dim, m);
        for (int j = 0; j < m; j++) {
            double[] diff = new double[dim];
            double[] v = trainingVectors.get(j);
            for (int i = 0; i < dim; i++) diff[i] = v[i] - mean[i];
            A.setColumn(j, diff);
        }

        // 1.3 - 1.6. Decomposição Eigen e obtenção de autovetores
        EigenDecomposition ed_pca = new EigenDecomposition(A.transpose().multiply(A)); // C = A^T * A

        double[] ev_pca = ed_pca.getRealEigenvalues();
        List<Integer> idx = new ArrayList<>();
        for (int i = 0; i < m; i++) idx.add(i);
        idx.sort((i, j) -> Double.compare(ev_pca[j], ev_pca[i])); // Ordenar decrescente

        List<RealVector> pcaEigenvectors = new ArrayList<>();
        for (int i : idx) {
            if (ev_pca[i] > 1e-12) {
                RealVector vSmall = ed_pca.getEigenvector(i);
                RealVector u = A.operate(vSmall);
                pcaEigenvectors.add(u.mapDivide(u.getNorm()));
            }
        }

        // 1.7. Criar a Matriz de Projeção PCA (W_pca)
        int k_pca_actual = Math.min(k_pca, pcaEigenvectors.size());
        if (k_pca_actual == 0) return null; // Será tratado no método train

        this.w_pca = new Array2DRowRealMatrix(dim, k_pca_actual);
        for (int c = 0; c < k_pca_actual; c++) {
            this.w_pca.setColumnVector(c, pcaEigenvectors.get(c));
        }

        // 1.8. Projetar todos os dados no espaço PCA
        Map<String, List<RealVector>> projectedClasses = new HashMap<>();
        for (String label : classes.keySet()) {
            List<RealVector> projectedVectors = new ArrayList<>();
            for (double[] v : classes.get(label)) {
                RealVector diff = new ArrayRealVector(v).subtract(this.meanFace);
                RealVector pcaCoeffs = this.w_pca.transpose().operate(diff);
                projectedVectors.add(pcaCoeffs);
            }
            projectedClasses.put(label, projectedVectors);
        }
        return projectedClasses;
    }

    private void executarLDA(Map<String, List<RealVector>> projectedClasses, int k_pca_actual, int k_lda, int m) {

        // 2.1. Calcular médias (no espaço PCA)
        RealVector globalMean_pca = new ArrayRealVector(k_pca_actual);
        Map<String, RealVector> classMeans_pca = new HashMap<>();

        for (String label : projectedClasses.keySet()) {
            RealVector classMean = new ArrayRealVector(k_pca_actual);
            List<RealVector> vectors = projectedClasses.get(label);
            for (RealVector v : vectors) {
                classMean = classMean.add(v);
                globalMean_pca = globalMean_pca.add(v);
            }
            classMeans_pca.put(label, classMean.mapDivide(vectors.size()));
        }
        globalMean_pca = globalMean_pca.mapDivide(m);

        // 2.2. Calcular Matriz Sw (Intra-classe)
        RealMatrix Sw = new Array2DRowRealMatrix(k_pca_actual, k_pca_actual);
        for (String label : projectedClasses.keySet()) {
            RealVector m_i = classMeans_pca.get(label);
            for (RealVector x : projectedClasses.get(label)) {
                RealVector diff = x.subtract(m_i);
                Sw = Sw.add(diff.outerProduct(diff));
            }
        }

        // 2.3. Calcular Matriz Sb (Inter-classe)
        RealMatrix Sb = new Array2DRowRealMatrix(k_pca_actual, k_pca_actual);
        for (String label : projectedClasses.keySet()) {
            int Ni = projectedClasses.get(label).size();
            RealVector m_i = classMeans_pca.get(label);
            RealVector diff = m_i.subtract(globalMean_pca);
            Sb = Sb.add(diff.outerProduct(diff).scalarMultiply(Ni));
        }

        // 2.4. Resolver (Sw^-1 * Sb)
        RealMatrix Sw_inv = new LUDecomposition(Sw).getSolver().getInverse();
        RealMatrix target = Sw_inv.multiply(Sb);

        // 2.5. Obter autovetores do LDA
        EigenDecomposition ed_lda = new EigenDecomposition(target);
        List<RealVector> ldaEigenvectors = new ArrayList<>();
        for (int i = 0; i < k_pca_actual; i++) {
            if (ed_lda.getRealEigenvalue(i) > 1e-12) {
                ldaEigenvectors.add(ed_lda.getEigenvector(i));
            }
        }

        // Ordenar por autovalor (decrescente)
        ldaEigenvectors.sort((v1, v2) -> {
            double ev1 = target.operate(v1).dotProduct(v1);
            double ev2 = target.operate(v2).dotProduct(v2);
            return Double.compare(ev2, ev1);
        });

        // 2.6. Criar a Matriz de Projeção LDA (W_lda)
        int k_lda_actual = Math.min(k_lda, ldaEigenvectors.size());
        this.w_lda = new Array2DRowRealMatrix(k_pca_actual, k_lda_actual);
        for (int c = 0; c < k_lda_actual; c++) {
            this.w_lda.setColumnVector(c, ldaEigenvectors.get(c));
        }
    }

    /**
     * Projeta os dados de treinamento (já no espaço PCA) para o espaço final (LDA).
     */
    private void projetarDadosFinais(Map<String, List<RealVector>> projectedClasses) {
        this.projectedFaces.clear();
        this.labels.clear();

        for (String label : projectedClasses.keySet()) {
            for (RealVector pca_vector : projectedClasses.get(label)) {
                // Projeta do espaço PCA (k_pca) para o espaço LDA (k_lda)
                RealVector final_coeffs = this.w_lda.transpose().operate(pca_vector);

                this.projectedFaces.add(final_coeffs.toArray());
                this.labels.add(label);
            }
        }
    }


    public RealMatrix getEigenfaces() { return eigenfaces; }
    public double[] getMeanVector() { return this.meanFace != null ? this.meanFace.toArray() : null; }
    public List<double[]> getProjectedFaces() { return projectedFaces; }
    public List<String> getLabels() { return labels; }
}