package FisherfacesModel;

import org.apache.commons.math3.linear.*;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

public class FisherfacesModel {

    // O modelo final (W_final = W_pca * W_lda)
    private RealMatrix eigenfaces;

    private RealVector meanFace;   // A Face Média Global (Ψ)
    private final List<double[]> projectedFaces; // Projeções FINAIS (no espaço LDA)
    private final List<String> labels; // Rótulos/Nomes associados

    public FisherfacesModel() {
        this.projectedFaces = new ArrayList<>();
        this.labels = new ArrayList<>();
    }

    /**
     * Implementa a Fase de Treinamento: PCA para redução, seguido por LDA para separação.
     * @param trainingVectors Lista de vetores de imagens de treinamento.
     * @param faceLabels Rótulos/Nomes correspondentes.
     */
    public void train(List<double[]> trainingVectors, List<String> faceLabels) {
        int m = trainingVectors.size(); // N (Número total de amostras)
        if (m == 0) throw new IllegalArgumentException("Nenhuma imagem de treinamento.");
        int dim = trainingVectors.get(0).length;

        // --- Agrupar dados por classe (Indivíduo) ---
        Map<String, List<double[]>> classes = new HashMap<>();
        for (int i = 0; i < m; i++) {
            classes.computeIfAbsent(faceLabels.get(i), k -> new ArrayList<>()).add(trainingVectors.get(i));
        }
        int C = classes.size(); // C (Número de classes/indivíduos)
        if (C <= 1) throw new IllegalArgumentException("O treinamento LDA requer pelo menos 2 classes (indivíduos).");

        // --- PASSO 1: LÓGICA DO PCA (para Redução de Dimensão) ---

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

        // 1.3. Calcular matriz pequena C = A^T * A (m x m)
        RealMatrix C_small = A.transpose().multiply(A);

        // 1.4. Decomposição Eigen de C
        EigenDecomposition ed_pca = new EigenDecomposition(C_small);

        // 1.5. Definir o K do PCA: k_pca = N - C
        int k_pca = m - C;

        // 1.6. Obter os autovetores (Eigenfaces) do PCA

        // 1.6.1. Obter autovalores e criar lista de índices
        double[] ev_pca = ed_pca.getRealEigenvalues();
        List<Integer> idx = new ArrayList<>();
        for (int i = 0; i < m; i++) idx.add(i); // m = ev_pca.length

        // 1.6.2. Ordenar índices com base nos autovalores (decrescente)
        // Isto garante que obtemos os vetores principais PRIMEIRO
        idx.sort((i, j) -> Double.compare(ev_pca[j], ev_pca[i]));

        // 1.6.3. Iterar na ordem correta (do maior autovalor para o menor)
        // e construir a lista pcaEigenvectors já ordenada
        List<RealVector> pcaEigenvectors = new ArrayList<>();
        for (int i : idx) { // Iteramos usando os índices ordenados
            double eigenvalue = ev_pca[i];

            if (eigenvalue > 1e-12) { // Ignorar autovalores muito pequenos
                RealVector vSmall = ed_pca.getEigenvector(i); // Vetor da matriz pequena (m x 1)
                RealVector u = A.operate(vSmall); // Converter para autovetor "grande" (dim x 1)
                // [dim x m] * [m x 1] = [dim x 1] -> VÁLIDO
                pcaEigenvectors.add(u.mapDivide(u.getNorm())); // Normalizar e adicionar
            }
        }
        // --- FIM DA CORREÇÃO ---

        // 1.7. Criar a Matriz de Projeção PCA (W_pca)
        // Pegamos os K_PCA (N-C) vetores principais
        int k_pca_actual = Math.min(k_pca, pcaEigenvectors.size());

        if (k_pca_actual == 0) {
            throw new RuntimeException(
                    String.format("PCA não gerou componentes. k_pca (N-C) = %d. Você precisa de mais imagens do que indivíduos (N > C).", k_pca)
            );
        }

        // Usar o campo da classe (this.W_pca)
        // --- CAMPOS DA CLASSE (Matrizes de Projeção) ---
        // Adicionei W_pca e W_lda que estavam em falta na sua versão
        // Projeção PCA (Redução de Dimensão)
        RealMatrix w_pca = new Array2DRowRealMatrix(dim, k_pca_actual);
        for (int c = 0; c < k_pca_actual; c++) {
            w_pca.setColumnVector(c, pcaEigenvectors.get(c));
        }

        // 1.8. Projetar todos os dados no espaço PCA
        Map<String, List<RealVector>> projectedClasses = new HashMap<>();
        for (String label : classes.keySet()) {
            List<RealVector> projectedVectors = new ArrayList<>();
            for (double[] v : classes.get(label)) {
                RealVector diff = new ArrayRealVector(v).subtract(this.meanFace);
                // Usar this.W_pca
                RealVector pcaCoeffs = w_pca.transpose().operate(diff); // Projeção
                projectedVectors.add(pcaCoeffs);
            }
            projectedClasses.put(label, projectedVectors);
        }

        // --- PASSO 2: LÓGICA DO LDA (Fisherfaces) ---
        // Agora, trabalhamos APENAS no espaço PCA (dimensão k_pca_actual)

        // 2.1. Calcular médias (no espaço PCA)

        // Média Global (no espaço PCA)
        RealVector globalMean_pca = new ArrayRealVector(k_pca_actual);
        for (List<RealVector> vectors : projectedClasses.values()) {
            for (RealVector v : vectors) {
                globalMean_pca = globalMean_pca.add(v);
            }
        }
        globalMean_pca = globalMean_pca.mapDivide(m);

        // Médias de Classe (no espaço PCA)
        Map<String, RealVector> classMeans_pca = new HashMap<>();
        for (String label : projectedClasses.keySet()) {
            RealVector classMean = new ArrayRealVector(k_pca_actual);
            List<RealVector> vectors = projectedClasses.get(label);
            for (RealVector v : vectors) {
                classMean = classMean.add(v);
            }
            classMeans_pca.put(label, classMean.mapDivide(vectors.size()));
        }

        // 2.2. Calcular Matriz de Dispersão Intra-classe (Sw)
        RealMatrix Sw = new Array2DRowRealMatrix(k_pca_actual, k_pca_actual);
        for (String label : projectedClasses.keySet()) {
            RealVector m_i = classMeans_pca.get(label);
            for (RealVector x : projectedClasses.get(label)) {
                RealVector diff = x.subtract(m_i);
                Sw = Sw.add(diff.outerProduct(diff)); // Sw = soma( (x - m_i) * (x - m_i)^T )
            }
        }

        // 2.3. Calcular Matriz de Dispersão Inter-classe (Sb)
        RealMatrix Sb = new Array2DRowRealMatrix(k_pca_actual, k_pca_actual);
        for (String label : projectedClasses.keySet()) {
            int Ni = projectedClasses.get(label).size(); // N. de amostras na classe
            RealVector m_i = classMeans_pca.get(label);
            RealVector diff = m_i.subtract(globalMean_pca);
            Sb = Sb.add(diff.outerProduct(diff).scalarMultiply(Ni)); // Sb = soma( Ni * (m_i - m) * (m_i - m)^T )
        }

        // 2.4. Resolver o Problema de Autovetor Generalizado: (Sw^-1 * Sb) * v = lambda * v
        RealMatrix Sw_inv = new LUDecomposition(Sw).getSolver().getInverse();
        RealMatrix target = Sw_inv.multiply(Sb);

        EigenDecomposition ed_lda = new EigenDecomposition(target);

        // 2.5. Obter os K_LDA (C-1) melhores autovetores
        int k_lda = C - 1; // Número máximo de componentes discriminantes

        List<RealVector> ldaEigenvectors = new ArrayList<>();
        for (int i = 0; i < k_pca_actual; i++) { // Iteramos no espaço PCA
            double eigenvalue = ed_lda.getRealEigenvalue(i);
            if (eigenvalue > 1e-12) {
                ldaEigenvectors.add(ed_lda.getEigenvector(i));
            }
        }

        // Ordenar por autovalor (decrescente)
        ldaEigenvectors.sort((v1, v2) -> {
            // Precisamos calcular o autovalor real (lambda) para (Sw^-1 * Sb)v = lambda*v
            double ev1 = target.operate(v1).dotProduct(v1);
            double ev2 = target.operate(v2).dotProduct(v2);
            return Double.compare(ev2, ev1);
        });

        // 2.6. Criar a Matriz de Projeção LDA (W_lda)
        int k_lda_actual = Math.min(k_lda, ldaEigenvectors.size());
        // Usar o campo da classe (this.W_lda)
        // Projeção LDA (Separação de Classe)
        RealMatrix w_lda = new Array2DRowRealMatrix(k_pca_actual, k_lda_actual);
        for (int c = 0; c < k_lda_actual; c++) {
            w_lda.setColumnVector(c, ldaEigenvectors.get(c));
        }

        // --- PASSO 3: FINALIZAÇÃO ---

        // 3.1. Criar a matriz de projeção final (Fisherface)
        // W_final = W_pca * W_lda
        // Esta é a matriz que o FaceRecognizer usará.
        this.eigenfaces = w_pca.multiply(w_lda);

        // 3.2. Projetar todas as faces de TREINAMENTO no espaço FINAL (LDA)
        this.projectedFaces.clear();
        this.labels.clear();

        for (String label : projectedClasses.keySet()) {
            // Re-usa os vetores já projetados pelo PCA
            for (RealVector pca_vector : projectedClasses.get(label)) {
                // Projeta do espaço PCA (k_pca) para o espaço LDA (k_lda)
                // Usar this.W_lda
                RealVector final_coeffs = w_lda.transpose().operate(pca_vector);

                this.projectedFaces.add(final_coeffs.toArray());
                this.labels.add(label);
            }
        }
    }

    // --- Getters (Sem alterações) ---

    public RealMatrix getEigenfaces() {
        return eigenfaces;
    }

    public double[] getMeanVector() {
        return this.meanFace != null ? this.meanFace.toArray() : null;
    }

    public List<double[]> getProjectedFaces() {
        return projectedFaces;
    }

    public List<String> getLabels() {
        return labels;
    }
}