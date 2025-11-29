package ImageProcessor;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class ImageProcessor {

    public static final int IMAGE_WIDTH = 100;
    public static final int IMAGE_HEIGHT = 100;
    public static final int VECTOR_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;

    /**
     * Lê uma imagem, redimensiona, aplica equalização de histograma e vetoriza.
     */
    public double[] processImage(File imageFile) throws IOException {
        BufferedImage originalImage = ImageIO.read(imageFile);
        if (originalImage == null) throw new IOException("Arquivo não é uma imagem válida: " + imageFile.getName());

        BufferedImage resizedImage = resizeImage(originalImage, IMAGE_WIDTH, IMAGE_HEIGHT);

        // Passo extra: Equalização de Histograma para lidar com iluminação
        double[] grayscale = convertToGrayscaleVector(resizedImage);
        return equalizeHistogram(grayscale);
    }

    private double[] convertToGrayscaleVector(BufferedImage img) {
        double[] vector = new double[VECTOR_SIZE];
        int idx = 0;
        for (int y = 0; y < IMAGE_HEIGHT; y++) {
            for (int x = 0; x < IMAGE_WIDTH; x++) {
                int rgb = img.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                // Fórmula de luminância padrão
                double gray = 0.299 * r + 0.587 * g + 0.114 * b;
                vector[idx++] = gray;
            }
        }
        return vector;
    }

    /**
     * Aplica equalização de histograma diretamente no vetor de pixels.
     */
    private double[] equalizeHistogram(double[] data) {
        int[] histogram = new int[256];
        for (double v : data) {
            histogram[(int) Math.min(255, Math.max(0, v))]++;
        }

        double[] cdf = new double[256];
        cdf[0] = histogram[0];
        for (int i = 1; i < 256; i++) {
            cdf[i] = cdf[i - 1] + histogram[i];
        }

        double minCdf = 0;
        for (int i = 0; i < 256; i++) {
            if (cdf[i] > 0) {
                minCdf = cdf[i];
                break;
            }
        }

        double totalPixels = data.length;
        double[] result = new double[data.length];

        for (int i = 0; i < data.length; i++) {
            int val = (int) Math.min(255, Math.max(0, data[i]));
            // Fórmula de equalização: h(v) = round(((cdf(v) - cdfMin) / (L - 1)) * 255)
            // Aqui usamos (total - 1) como denominador da escala
            double newVal = ((cdf[val] - minCdf) / (totalPixels - 1)) * 255.0;
            result[i] = newVal;
        }
        return result;
    }

    private BufferedImage resizeImage(BufferedImage originalImage, int width, int height) {
        Image tmp = originalImage.getScaledInstance(width, height, Image.SCALE_SMOOTH);
        BufferedImage resizedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resizedImage.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();
        return resizedImage;
    }
}