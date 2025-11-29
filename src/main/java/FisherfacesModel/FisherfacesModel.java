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

        System.out.println("\n=== INÍCIO DO TREINAMENTO MATEMÁTICO ===");
        System.out.printf("Dados de Entrada: %d imagens com %d pixels (dimensões) cada.%n", m, dim);

        Map<String, List<double[]>> classes = agruparPorClasse(data.vectors(), data.labels());
        int C = classes.size();
        if (C <= 1) throw new IllegalArgumentException("O treinamento LDA requer pelo menos 2 classes (indivíduos).");

        // --- PASSO 1: PCA (Principal Component Analysis) ---
        // Objetivo: Reduzir ruído e dimensionalidade bruta
        int k_pca = m - C;
        System.out.println("\n--- PASSO 1: PCA (Análise de Componentes Principais) ---");
        System.out.println("Objetivo: Encontrar as direções de maior variação global nos rostos.");

        Map<String, List<RealVector>> projectedClasses = executarPCA(data.vectors(), classes, dim, m, k_pca);

        if (this.w_pca == null) throw new RuntimeException("PCA falhou.");
        int k_pca_actual = this.w_pca.getColumnDimension();
        System.out.printf("Resultado PCA: Reduzido de %d dimensões para %d características principais.%n", dim, k_pca_actual);

        // --- PASSO 2: LDA (Linear Discriminant Analysis) ---
        // Objetivo: Classificação (Separar classes)
        int k_lda = C - 1;
        System.out.println("\n--- PASSO 2: LDA (Análise Discriminante Linear) ---");
        System.out.println("Objetivo: Maximizar distância entre pessoas diferentes e minimizar variação da mesma pessoa.");

        executarLDA(projectedClasses, k_pca_actual, k_lda, m);

        // --- PASSO 3: Finalização ---
        this.eigenfaces = this.w_pca.multiply(this.w_lda);
        projetarDadosFinais(projectedClasses);
        System.out.println("\n=== TREINAMENTO CONCLUÍDO ===");
        System.out.printf("Dimensão Final do Espaço de Faces: %d (suficiente para distinguir %d pessoas)%n",
                this.eigenfaces.getColumnDimension(), C);
    }

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

        RealMatrix covariance = A.transpose().multiply(A);
        EigenDecomposition ed_pca = new EigenDecomposition(covariance);

        double[] ev_pca = ed_pca.getRealEigenvalues();

        // MOSTRAR AUTOVALORES (IMPORTÂNCIA DAS CARACTERÍSTICAS)
        System.out.print("  [Didático] Top 5 Autovalores (Importância): ");
        for(int i=0; i<Math.min(5, ev_pca.length); i++) System.out.printf("%.2e; ", ev_pca[i]);
        System.out.println("...");

        List<Integer> idx = new ArrayList<>();
        for (int i = 0; i < m; i++) idx.add(i);
        idx.sort((i, j) -> Double.compare(ev_pca[j], ev_pca[i]));

        List<RealVector> pcaEigenvectors = new ArrayList<>();
        for (int i : idx) {
            if (ev_pca[i] > 1e-10) {
                RealVector vSmall = ed_pca.getEigenvector(i);
                RealVector u = A.operate(vSmall);
                u = u.mapDivide(u.getNorm());
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

        // 2.2. Sw (Matriz de Dispersão Intra-classe)
        // TEORIA: Queremos MINIMIZAR isto. Representa o quanto a foto do "João" difere de outra foto do "João".
        RealMatrix Sw = new Array2DRowRealMatrix(k_pca_actual, k_pca_actual);
        for (String label : projectedClasses.keySet()) {
            RealVector m_i = classMeans_pca.get(label);
            for (RealVector x : projectedClasses.get(label)) {
                RealVector diff = x.subtract(m_i);
                Sw = Sw.add(diff.outerProduct(diff));
            }
        }
        System.out.printf("  [Matemática] Matriz Sw (Intra-classe) calculada [%dx%d]. Representa variação interna.%n", k_pca_actual, k_pca_actual);

        RealMatrix identity = MatrixUtils.createRealIdentityMatrix(k_pca_actual);
        Sw = Sw.add(identity.scalarMultiply(1e-5)); // Regularização

        // 2.3. Sb (Matriz de Dispersão Inter-classe)
        // TEORIA: Queremos MAXIMIZAR isto. Representa a distância média entre o "João" e a "Maria".
        RealMatrix Sb = new Array2DRowRealMatrix(k_pca_actual, k_pca_actual);
        for (String label : projectedClasses.keySet()) {
            int Ni = projectedClasses.get(label).size();
            RealVector m_i = classMeans_pca.get(label);
            RealVector diff = m_i.subtract(globalMean_pca);
            Sb = Sb.add(diff.outerProduct(diff).scalarMultiply(Ni));
        }
        System.out.printf("  [Matemática] Matriz Sb (Inter-classe) calculada [%dx%d]. Representa separação entre pessoas.%n", k_pca_actual, k_pca_actual);

        // 2.4. Resolver problema generalizado de autovalores: Sw^-1 * Sb
        RealMatrix Sw_inv = new LUDecomposition(Sw).getSolver().getInverse();
        RealMatrix target = Sw_inv.multiply(Sb);

        EigenDecomposition ed_lda = new EigenDecomposition(target);

        List<Integer> indices = new ArrayList<>();
        for(int i=0; i < k_pca_actual; i++) indices.add(i);
        indices.sort((i, j) -> Double.compare(ed_lda.getRealEigenvalue(j), ed_lda.getRealEigenvalue(i)));

        int k_lda_actual = Math.min(k_lda, k_pca_actual);

        this.w_lda = new Array2DRowRealMatrix(k_pca_actual, k_lda_actual);
        int col = 0;
        for (int i : indices) {
            if(col >= k_lda_actual) break;
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