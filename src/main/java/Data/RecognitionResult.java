package Data;

/**
 * Encapsula o resultado de uma única verificação de suspeito.
 */
public record RecognitionResult(
    String fileName,
    String recognizedLabel,
    double distance,
    boolean isMatch
) {
    @Override
    public String toString() {
        if (isMatch) {
            return String.format("ALERTA! [Arquivo: %s]: CORRESPONDÊNCIA ENCONTRADA! -> %s (Distância: %.2f)",
                    fileName, recognizedLabel, distance);
        } else {
            return String.format("VERIFICANDO [Arquivo: %s]: NÃO ENCONTRADO. (Mais próximo: %s, Distância: %.2f)",
                    fileName, recognizedLabel, distance);
        }
    }
}
