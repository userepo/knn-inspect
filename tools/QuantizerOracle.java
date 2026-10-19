// Runs Lucene's own OptimizedScalarQuantizer over the cases
// tools/quantizer_oracle.py writes, as the expected results for the Rust
// port. Run with Lucene's core jar on the class path, in source-file
// mode:
//
//   java -cp lucene-core-10.5.1.jar QuantizerOracle.java <cases> <expected>
//
// <cases>: a little-endian uint32 count, then per case the similarity
// ordinal (uint8), the bits (uint8), the dimension (uint32), the vector
// and the centroid (float32 each).
// <expected>: per case, one code per component (uint8), then the lower
// and upper interval ends and the additional correction (float32 each)
// and the component sum (int32), little-endian.

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

public class QuantizerOracle {
  public static void main(String[] args) throws IOException {
    ByteBuffer in =
        ByteBuffer.wrap(Files.readAllBytes(Path.of(args[0]))).order(ByteOrder.LITTLE_ENDIAN);
    int count = in.getInt();
    ByteArrayOutputStream expected = new ByteArrayOutputStream();
    for (int c = 0; c < count; c++) {
      VectorSimilarityFunction similarity = VectorSimilarityFunction.values()[in.get()];
      byte bits = in.get();
      int dimension = in.getInt();
      float[] vector = new float[dimension];
      float[] centroid = new float[dimension];
      for (int i = 0; i < dimension; i++) {
        vector[i] = in.getFloat();
      }
      for (int i = 0; i < dimension; i++) {
        centroid[i] = in.getFloat();
      }
      byte[] codes = new byte[dimension];
      OptimizedScalarQuantizer.QuantizationResult r =
          new OptimizedScalarQuantizer(similarity).scalarQuantize(vector, codes, bits, centroid);
      ByteBuffer out = ByteBuffer.allocate(dimension + 16).order(ByteOrder.LITTLE_ENDIAN);
      out.put(codes);
      out.putFloat(r.lowerInterval());
      out.putFloat(r.upperInterval());
      out.putFloat(r.additionalCorrection());
      out.putInt(r.quantizedComponentSum());
      expected.write(out.array());
    }
    Files.write(Path.of(args[1]), expected.toByteArray());
  }
}
