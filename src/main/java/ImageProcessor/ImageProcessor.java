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

    public double[] processImage(File imageFile) throws IOException {
        BufferedImage originalImage = ImageIO.read(imageFile);
        if (originalImage == null) throw new IOException("Arquivo não é uma imagem válida: " + imageFile.getName());

        BufferedImage resizedImage = resizeImage(originalImage, IMAGE_WIDTH, IMAGE_HEIGHT);
        double[] grayscale = convertToGrayscaleVector(resizedImage);
        return equalizeHistogram(grayscale);
    }

    /**
     * NOVO: Converte um vetor matemático (ex: Eigenface) numa imagem PNG para visualização.
     * Normaliza os valores para 0-255.
     */
    public void saveVectorAsImage(double[] vector, String outputPath) {
        if (vector.length != VECTOR_SIZE) {
            System.err.println("Tamanho do vetor incorreto para imagem: " + vector.length);
            return;
        }

        // 1. Encontrar Min e Max para normalização
        BufferedImage image = getBufferedImage(vector);

        try {
            File outputFile = new File(outputPath);
            // Cria diretoria pai se não existir
            if (outputFile.getParentFile() != null) outputFile.getParentFile().mkdirs();
            ImageIO.write(image, "png", outputFile);
            System.out.println("  -> Imagem gerada: " + outputPath);
        } catch (IOException e) {
            System.err.println("Erro ao salvar imagem do vetor: " + e.getMessage());
        }
    }

    private static BufferedImage getBufferedImage(double[] vector) {
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        for (double v : vector) {
            if (v < min) min = v;
            if (v > max) max = v;
        }

        BufferedImage image = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT, BufferedImage.TYPE_INT_RGB);
        int idx = 0;
        for (int y = 0; y < IMAGE_HEIGHT; y++) {
            for (int x = 0; x < IMAGE_WIDTH; x++) {
                double val = vector[idx++];
                // Mapeamento linear de [min, max] para [0, 255]
                int gray = (int) ((val - min) / (max - min) * 255.0);
                // Clamp para garantir segurança
                gray = Math.min(255, Math.max(0, gray));

                int rgb = (gray << 16) | (gray << 8) | gray;
                image.setRGB(x, y, rgb);
            }
        }
        return image;
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
                double gray = 0.299 * r + 0.587 * g + 0.114 * b;
                vector[idx++] = gray;
            }
        }
        return vector;
    }

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