package Data;

import java.util.List;

public record TrainingData(
    List<double[]> vectors,
    List<String> labels
) {
    public int size() {
        return vectors.size();
    }

    public boolean isEmpty() {
        return vectors.isEmpty();
    }
}
