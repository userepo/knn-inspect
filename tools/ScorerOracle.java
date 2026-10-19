// Runs Lucene's own Lucene104ScalarQuantizedVectorScorer over the cases
// tools/scorer_oracle.py writes, as the expected scores for the Rust port.
// Run with Lucene's core jar on the class path, in source-file mode:
//
//   java -cp lucene-core-10.5.1.jar ScorerOracle.java <cases> <expected>
//
// <cases>: a little-endian uint32 count, then per case the similarity
// ordinal (uint8), the scalar encoding's wire number (uint8), the
// dimension, the number of stored vectors, and the number of queries
// (uint32 each), the centroid's dot product with itself (float32), the
// centroid, the stored vectors, and the queries (float32 each).
// <expected>: per case, per query, the score of every stored vector
// (float32, little-endian). Lucene quantizes the stored vectors itself,
// as its writer does, before scoring them.

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;

public class ScorerOracle {
  /** Stored vectors as Lucene's reader presents them, from arrays. */
  static final class Stored extends QuantizedByteVectorValues {
    final byte[][] packed;
    final OptimizedScalarQuantizer.QuantizationResult[] terms;
    final ScalarEncoding encoding;
    final float[] centroid;
    final float centroidDp;
    final OptimizedScalarQuantizer quantizer;
    final int dimension;

    Stored(
        byte[][] packed,
        OptimizedScalarQuantizer.QuantizationResult[] terms,
        ScalarEncoding encoding,
        float[] centroid,
        float centroidDp,
        OptimizedScalarQuantizer quantizer,
        int dimension) {
      this.packed = packed;
      this.terms = terms;
      this.encoding = encoding;
      this.centroid = centroid;
      this.centroidDp = centroidDp;
      this.quantizer = quantizer;
      this.dimension = dimension;
    }

    @Override
    public OptimizedScalarQuantizer.QuantizationResult getCorrectiveTerms(int ord) {
      return terms[ord];
    }

    @Override
    public OptimizedScalarQuantizer getQuantizer() {
      return quantizer;
    }

    @Override
    public ScalarEncoding getScalarEncoding() {
      return encoding;
    }

    @Override
    public float[] getCentroid() {
      return centroid;
    }

    @Override
    public float getCentroidDP() {
      return centroidDp;
    }

    @Override
    public Stored copy() {
      return this;
    }

    @Override
    public byte[] vectorValue(int ord) {
      return packed[ord];
    }

    @Override
    public int dimension() {
      return dimension;
    }

    @Override
    public int size() {
      return packed.length;
    }
  }

  /** Packs one code per component as Lucene's writer does. */
  static byte[] pack(ScalarEncoding encoding, byte[] codes, int dimension) {
    byte[] packed = new byte[encoding.getDocPackedLength(dimension)];
    switch (encoding) {
      case UNSIGNED_BYTE, SEVEN_BIT -> System.arraycopy(codes, 0, packed, 0, codes.length);
      // OffHeapScalarQuantizedVectorValues.packNibbles, which is not public.
      case PACKED_NIBBLE -> {
        for (int i = 0; i < packed.length; i++) {
          packed[i] = (byte) (codes[i] << 4 | codes[packed.length + i]);
        }
      }
      case SINGLE_BIT_QUERY_NIBBLE -> OptimizedScalarQuantizer.packAsBinary(codes, packed);
      case DIBIT_QUERY_NIBBLE -> OptimizedScalarQuantizer.transposeDibit(codes, packed);
    }
    return packed;
  }

  static float[] floats(ByteBuffer in, int n) {
    float[] out = new float[n];
    for (int i = 0; i < n; i++) {
      out[i] = in.getFloat();
    }
    return out;
  }

  public static void main(String[] args) throws IOException {
    ByteBuffer in =
        ByteBuffer.wrap(Files.readAllBytes(Path.of(args[0]))).order(ByteOrder.LITTLE_ENDIAN);
    int count = in.getInt();
    ByteArrayOutputStream expected = new ByteArrayOutputStream();
    for (int c = 0; c < count; c++) {
      VectorSimilarityFunction similarity = VectorSimilarityFunction.values()[in.get()];
      ScalarEncoding encoding = ScalarEncoding.fromWireNumber(in.get()).orElseThrow();
      int dimension = in.getInt();
      int docs = in.getInt();
      int queries = in.getInt();
      float centroidDp = in.getFloat();
      float[] centroid = floats(in, dimension);
      OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(similarity);
      byte[][] packed = new byte[docs][];
      OptimizedScalarQuantizer.QuantizationResult[] terms =
          new OptimizedScalarQuantizer.QuantizationResult[docs];
      for (int d = 0; d < docs; d++) {
        byte[] codes = new byte[encoding.getDiscreteDimensions(dimension)];
        terms[d] =
            quantizer.scalarQuantize(floats(in, dimension), codes, encoding.getBits(), centroid);
        packed[d] = pack(encoding, codes, dimension);
      }
      Stored stored =
          new Stored(packed, terms, encoding, centroid, centroidDp, quantizer, dimension);
      Lucene104ScalarQuantizedVectorScorer scorer = new Lucene104ScalarQuantizedVectorScorer(null);
      ByteBuffer out = ByteBuffer.allocate(4 * docs * queries).order(ByteOrder.LITTLE_ENDIAN);
      for (int q = 0; q < queries; q++) {
        RandomVectorScorer s =
            scorer.getRandomVectorScorer(similarity, stored, floats(in, dimension));
        for (int d = 0; d < docs; d++) {
          out.putFloat(s.score(d));
        }
      }
      expected.write(out.array());
    }
    Files.write(Path.of(args[1]), expected.toByteArray());
  }
}
