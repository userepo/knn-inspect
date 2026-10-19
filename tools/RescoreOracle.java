// Runs Lucene's float32 vector sums as k-NN 3.9 rescores with them, over the
// cases tools/rescore_oracle.py writes, as the expected values for the Rust
// port (knn_recall::on_disk::exact_score and knn_recall::quantizer::
// l2normalize). Run with Lucene's core jar on the class path, in
// source-file mode, with the JDK's vector module, as OpenSearch runs
// Lucene:
//
//   java --add-modules jdk.incubator.vector -cp lucene-core-10.5.1.jar \
//       RescoreOracle.java <cases> <expected>
//
// <cases>: a little-endian uint32 count, then per case the similarity
// ordinal (uint8), the dimension, the number of stored vectors, and the
// number of queries (uint32 each), the stored vectors, and the queries
// (float32 each).
// <expected>: per case, per query, the query as VectorUtil.l2normalize
// leaves it, then the score of every stored vector against the query (the
// normalized one for COSINE), from the rescorer of the stored vectors in a
// memory-mapped file, in bulk, as k-NN's ExactSearcher scores them
// (float32, little-endian).
//
// Compiled, Lucene's vector code adds its SIMD lanes in another order than
// interpreted, so everything runs until the JVM has compiled it before
// anything is recorded; it is then recorded twice, and must not change.

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocAndFloatFeatureBuffer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.VectorUtil;

public class RescoreOracle {
  static final int WARM_UP_ROUNDS = 200;

  record Case(
      VectorSimilarityFunction similarity,
      int dimension,
      OffHeapFloatVectorValues stored,
      float[][] queries) {}

  public static void main(String[] args) throws IOException {
    ByteBuffer in =
        ByteBuffer.wrap(Files.readAllBytes(Path.of(args[0]))).order(ByteOrder.LITTLE_ENDIAN);
    Path dir = Files.createTempDirectory("rescore-oracle");
    try (Directory directory = new MMapDirectory(dir)) {
      Case[] cases = new Case[in.getInt()];
      for (int c = 0; c < cases.length; c++) {
        VectorSimilarityFunction similarity =
            VectorSimilarityFunction.values()[Byte.toUnsignedInt(in.get())];
        int dimension = in.getInt();
        int count = in.getInt();
        int queryCount = in.getInt();
        String name = "case" + c;
        try (IndexOutput out = directory.createOutput(name, IOContext.DEFAULT)) {
          for (int i = 0; i < count * dimension; i++) {
            out.writeInt(Float.floatToRawIntBits(in.getFloat()));
          }
        }
        IndexInput input = directory.openInput(name, IOContext.DEFAULT);
        OffHeapFloatVectorValues stored =
            new OffHeapFloatVectorValues.DenseOffHeapVectorValues(
                dimension,
                count,
                input,
                dimension * Float.BYTES,
                FlatVectorScorerUtil.getLucene99FlatVectorsScorer(),
                similarity);
        float[][] queries = new float[queryCount][dimension];
        for (float[] q : queries) {
          for (int i = 0; i < dimension; i++) {
            q[i] = in.getFloat();
          }
        }
        cases[c] = new Case(similarity, dimension, stored, queries);
      }
      for (int round = 0; round < WARM_UP_ROUNDS; round++) {
        run(cases);
      }
      byte[] expected = run(cases);
      if (!Arrays.equals(expected, run(cases))) {
        throw new IllegalStateException("the sums changed after the warm-up");
      }
      Files.write(Path.of(args[1]), expected);
    }
  }

  static byte[] run(Case[] cases) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    for (Case c : cases) {
      for (float[] query : c.queries) {
        float[] normalized = VectorUtil.l2normalize(query.clone(), false);
        float[] target = c.similarity == VectorSimilarityFunction.COSINE ? normalized : query;
        float[] scores = new float[c.stored.size()];
        VectorScorer.Bulk bulk = c.stored.rescorer(target).bulk(null);
        DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
        for (int seen = 0; seen < scores.length; seen += buffer.size) {
          bulk.nextDocsAndScores(DocIdSetIterator.NO_MORE_DOCS, null, buffer);
          if (buffer.size == 0) {
            throw new IllegalStateException("fewer scores than stored vectors");
          }
          for (int i = 0; i < buffer.size; i++) {
            scores[buffer.docs[i]] = buffer.features[i];
          }
        }
        write(out, normalized);
        write(out, scores);
      }
    }
    return out.toByteArray();
  }

  static void write(ByteArrayOutputStream out, float[] values) {
    ByteBuffer b = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float v : values) {
      b.putFloat(v);
    }
    out.writeBytes(b.array());
  }
}
