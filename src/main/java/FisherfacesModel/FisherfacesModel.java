package FisherfacesModel;

import Data.TrainingData;
import org.apache.commons.math3.linear.*;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

public class FisherfacesModel {

    private RealMatrix eigenfaces; // W_final
    private RealVector meanFace;
    private final List<double[]> projectedFaces;
    private final List<String> labels;

    // Matrizes de projeção intermediárias
    private RealMatrix w_pca;
    private RealMatrix w_lda;

    public FisherfacesModel() {
        this.projectedFaces = new ArrayList<>();
        this.labels = new ArrayList<>();
    }

    public void train(TrainingData data) {
        int m = data.size();
        if (m == 0) throw new IllegalArgumentException("Nenhuma imagem de treinamento.");
        int dim = data.vectors().getFirst().length;

        Map<String, List<double[]>> classes = agruparPorClasse(data.vectors(), data.labels());
        int C = classes.size();
        if (C <= 1) throw new IllegalArgumentException("O treinamento LDA requer pelo menos 2 classes (indivíduos).");

        // --- PASSO 1: PCA ---
        int k_pca = m - C;
        Map<String, List<RealVector>> projectedClasses = executarPCA(data.vectors(), classes, dim, m, k_pca);

        if (this.w_pca == null) {
            throw new RuntimeException("PCA falhou em gerar componentes suficientes.");
        }

        int k_pca_actual = this.w_pca.getColumnDimension();

        // --- PASSO 2: LDA ---
        int k_lda = C - 1;
        assert projectedClasses != null;
        executarLDA(projectedClasses, k_pca_actual, k_lda, m);

        // --- PASSO 3: Finalização ---
        this.eigenfaces = this.w_pca.multiply(this.w_lda);
        projetarDadosFinais(projectedClasses);
    }

    // --- MÉTODOS PRIVADOS ---

    private Map<String, List<double[]>> agruparPorClasse(List<double[]> trainingVectors, List<String> faceLabels) {
        Map<String, List<double[]>> classes = new HashMap<>();
        for (int i = 0; i < trainingVectors.size(); i++) {
            classes.computeIfAbsent(faceLabels.get(i), k -> new ArrayList<>()).add(trainingVectors.get(i));
        }
        return classes;
    }

    private Map<String, List<RealVector>> executarPCA(List<double[]> trainingVectors, Map<String, List<double[]>> classes, int dim, int m, int k_pca) {
        // 1.1. Calcular média global
        double[] mean = new double[dim];
        for (double[] v : trainingVectors) {
            for (int i = 0; i < dim; i++) mean[i] += v[i];
        }
        for (int i = 0; i < dim; i++) mean[i] /= m;
        this.meanFace = new ArrayRealVector(mean);

        // 1.2. Montar Matriz A centralizada
        RealMatrix A = new Array2DRowRealMatrix(dim, m);
        for (int j = 0; j < m; j++) {
            double[] v = trainingVectors.get(j);
            for (int i = 0; i < dim; i++) {
                A.setEntry(i, j, v[i] - mean[i]);
            }
        }

        // 1.3. Decomposição Eigen em A^T * A (Snapshot method)
        RealMatrix covariance = A.transpose().multiply(A);
        EigenDecomposition ed_pca = new EigenDecomposition(covariance);

        double[] ev_pca = ed_pca.getRealEigenvalues();
        List<Integer> idx = new ArrayList<>();
        for (int i = 0; i < m; i++) idx.add(i);
        idx.sort((i, j) -> Double.compare(ev_pca[j], ev_pca[i]));

        List<RealVector> pcaEigenvectors = new ArrayList<>();
        for (int i : idx) {
            if (ev_pca[i] > 1e-10) { // Tolerância ajustada
                RealVector vSmall = ed_pca.getEigenvector(i);
                RealVector u = A.operate(vSmall);
                u = u.mapDivide(u.getNorm()); // Normalização crucial
                pcaEigenvectors.add(u);
            }
        }

        int k_pca_actual = Math.min(k_pca, pcaEigenvectors.size());
        if (k_pca_actual == 0) return null;

        this.w_pca = new Array2DRowRealMatrix(dim, k_pca_actual);
        for (int c = 0; c < k_pca_actual; c++) {
            this.w_pca.setColumnVector(c, pcaEigenvectors.get(c));
        }

        Map<String, List<RealVector>> projectedClasses = new HashMap<>();
        for (String label : classes.keySet()) {
            List<RealVector> projectedVectors = new ArrayList<>();
            for (double[] v : classes.get(label)) {
                RealVector diff = new ArrayRealVector(v).subtract(this.meanFace);
                projectedVectors.add(this.w_pca.transpose().operate(diff));
            }
            projectedClasses.put(label, projectedVectors);
        }
        return projectedClasses;
    }

    private void executarLDA(Map<String, List<RealVector>> projectedClasses, int k_pca_actual, int k_lda, int m) {
        // 2.1. Médias PCA
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

        // 2.2. Sw (Intra-classe)
        RealMatrix Sw = new Array2DRowRealMatrix(k_pca_actual, k_pca_actual);
        for (String label : projectedClasses.keySet()) {
            RealVector m_i = classMeans_pca.get(label);
            for (RealVector x : projectedClasses.get(label)) {
                RealVector diff = x.subtract(m_i);
                Sw = Sw.add(diff.outerProduct(diff));
            }
        }

        // REGULARIZAÇÃO: Adiciona ruído na diagonal para evitar singularidade
        RealMatrix identity = MatrixUtils.createRealIdentityMatrix(k_pca_actual);
        Sw = Sw.add(identity.scalarMultiply(1e-5));

        // 2.3. Sb (Inter-classe)
        RealMatrix Sb = new Array2DRowRealMatrix(k_pca_actual, k_pca_actual);
        for (String label : projectedClasses.keySet()) {
            int Ni = projectedClasses.get(label).size();
            RealVector m_i = classMeans_pca.get(label);
            RealVector diff = m_i.subtract(globalMean_pca);
            Sb = Sb.add(diff.outerProduct(diff).scalarMultiply(Ni));
        }

        // 2.4. Resolver (Sw^-1 * Sb)
        // Usar solver LU é mais rápido e estável com a regularização
        RealMatrix Sw_inv = new LUDecomposition(Sw).getSolver().getInverse();
        RealMatrix target = Sw_inv.multiply(Sb);

        // 2.5. Autovetores LDA
        EigenDecomposition ed_lda = new EigenDecomposition(target);
        List<RealVector> ldaEigenvectors = new ArrayList<>();

        // Coleta e Ordenação
        List<Integer> indices = new ArrayList<>();
        for(int i=0; i < k_pca_actual; i++) indices.add(i);

        // Ordenar baseando-se nos autovalores REAIS (parte imaginária deve ser nula para matrizes simétricas/PSD)
        indices.sort((i, j) -> Double.compare(ed_lda.getRealEigenvalue(j), ed_lda.getRealEigenvalue(i)));

        int k_lda_actual = Math.min(k_lda, k_pca_actual);

        this.w_lda = new Array2DRowRealMatrix(k_pca_actual, k_lda_actual);
        int col = 0;
        for (int i : indices) {
            if(col >= k_lda_actual) break;
            // Ignorar autovalores muito pequenos ou negativos (ruído numérico)
            if (ed_lda.getRealEigenvalue(i) > 1e-12) {
                this.w_lda.setColumnVector(col++, ed_lda.getEigenvector(i));
            }
        }
    }

    private void projetarDadosFinais(Map<String, List<RealVector>> projectedClasses) {
        this.projectedFaces.clear();
        this.labels.clear();
        for (String label : projectedClasses.keySet()) {
            for (RealVector pca_vector : projectedClasses.get(label)) {
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