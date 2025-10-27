package ImageProcessor;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class ImageProcessor {

    // Tamanho das imagens a serem redimensionadas (Exemplo: 100x100 = 10000 pixels)
    public static final int IMAGE_WIDTH = 100;
    public static final int IMAGE_HEIGHT = 100;
    public static final int VECTOR_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;

    /**
     * Lê uma imagem, converte para escala de cinza e vetoriza.
     * @param imageFile O arquivo da imagem.
     * @return Um vetor (array de double) de pixels.
     */
    public double[] processImage(File imageFile) throws IOException {
        BufferedImage originalImage = ImageIO.read(imageFile);
        BufferedImage resizedImage = resizeImage(originalImage, IMAGE_WIDTH, IMAGE_HEIGHT);
        return convertToGrayscaleVector(resizedImage);
    }

    // [Implementação do método de Redimensionamento (java.awt.Image)]
    private BufferedImage resizeImage(BufferedImage originalImage, int width, int height) {
        // Implementação do redimensionamento aqui. Pode usar getScaledInstance ou similar.
        // **Nota:** A implementação exata de redimensionamento e conversão de cor
        // pode ser complexa e é onde uma biblioteca como JavaCV/OpenCV ajudaria.
        // Para fins deste projeto didático, assuma que esta função retorna a imagem
        // no tamanho correto ou use imagens pré-processadas.
        return originalImage; // Implementação simplificada
    }
    
    /**
     * Converte a imagem (redimensionada) para escala de cinza e para um vetor 1D de doubles.
     * @param img A imagem processada.
     * @return O vetor de pixels 1D.
     */
    private double[] convertToGrayscaleVector(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        int size = width * height;
        double[] vector = new double[size];
        int idx = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = img.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                double gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                vector[idx++] = gray;
            }
        }
        return vector;
    }
}
