// Runs k-NN 3.9's rescoring of a half_float field's fp16 vectors, over the
// cases tools/fp16_rescore_oracle.py writes, as the expected values for the
// Rust port (knn_recall::on_disk::fp16_exact_score). Run with the OpenSearch
// image's jars and the k-NN plugin's on the class path, its native libraries
// on the library path, in source-file mode, with the JDK's vector module, as
// OpenSearch runs k-NN:
//
//   java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
//       -Djava.library.path=plugins/opensearch-knn/lib \
//       -cp 'lib/*:plugins/opensearch-knn/*:modules/transport-grpc/guava-33.2.1-jre.jar' \
//       Fp16RescoreOracle.java <cases> <expected>
//
// <cases>: a little-endian uint32 count, then per case the similarity
// ordinal (uint8), the dimension, the number of stored vectors, and the
// number of queries (uint32 each), the stored vectors (fp16 each), and the
// queries (float32 each).
// <expected>: per case, per query, the score of every stored vector against
// the query (float32, little-endian), from the rescorer of k-NN's own fp16
// vector values over the stored vectors in a memory-mapped file, in bulk, as
// k-NN's ExactSearcher scores them: k-NN's native kernel for the similarities
// it has one for, Lucene's function over the decoded vectors for the others.
//
// The values are KNN1040HalfFloatFlatVectorsValues, k-NN's half_float
// format's, made through their package-private constructor. The native
// library is the one k-NN loads for the host's CPU; the oracle prints which,
// and tools/fp16_rescore_oracle.py refuses any but the AVX2 one, whose sums
// the Rust port follows. k-NN picks that library by its node settings, which
// it reads through the cluster service: the oracle gives it one of default
// settings. Which scorer k-NN chose is checked too, through its own
// selectScorer: the native one for every similarity it has a kernel for,
// its Java fallback for the others. Lucene's function compiles to another
// lane order than it runs interpreted, so everything runs until the JVM has
// compiled it before anything is recorded; it is then recorded twice, and
// must not change.

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocAndFloatFeatureBuffer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.KNNSettings;

public class Fp16RescoreOracle {
  static final int WARM_UP_ROUNDS = 200;

  record Case(
      VectorSimilarityFunction similarity,
      int dimension,
      FloatVectorValues stored,
      float[][] queries) {}

  public static void main(String[] args) throws Exception {
    // The node settings k-NN reads choosing its native library, at their
    // defaults, beside those the cluster service registers.
    Set<Setting<?>> nodeSettings =
        KNNSettings.state().getSettings().stream()
            .filter(Setting::hasNodeScope)
            .collect(Collectors.toCollection(HashSet::new));
    nodeSettings.addAll(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
    Field clusterService = KNNSettings.class.getDeclaredField("clusterService");
    clusterService.setAccessible(true);
    clusterService.set(
        KNNSettings.state(),
        new ClusterService(
            Settings.EMPTY, new ClusterSettings(Settings.EMPTY, nodeSettings), null));
    ByteBuffer in =
        ByteBuffer.wrap(Files.readAllBytes(Path.of(args[0]))).order(ByteOrder.LITTLE_ENDIAN);
    Class<?> valuesClass =
        Class.forName(
            "org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsValues");
    Constructor<?> values =
        valuesClass.getDeclaredConstructor(
            int.class,
            int.class,
            IndexInput.class,
            DirectMonotonicReader.class,
            VectorSimilarityFunction.class);
    values.setAccessible(true);
    Method selectScorer =
        valuesClass.getDeclaredMethod(
            "selectScorer", valuesClass, float[].class, VectorSimilarityFunction.class);
    selectScorer.setAccessible(true);
    Path dir = Files.createTempDirectory("fp16-rescore-oracle");
    List<IndexInput> inputs = new ArrayList<>();
    try (Directory directory = new MMapDirectory(dir)) {
      Case[] cases = new Case[in.getInt()];
      for (int c = 0; c < cases.length; c++) {
        VectorSimilarityFunction similarity =
            VectorSimilarityFunction.values()[Byte.toUnsignedInt(in.get())];
        int dimension = in.getInt();
        int count = in.getInt();
        int queryCount = in.getInt();
        String name = "case" + c;
        byte[] fp16 = new byte[count * dimension * Short.BYTES];
        in.get(fp16);
        try (IndexOutput out = directory.createOutput(name, IOContext.DEFAULT)) {
          out.writeBytes(fp16, fp16.length);
        }
        IndexInput input = directory.openInput(name, IOContext.DEFAULT);
        inputs.add(input);
        FloatVectorValues stored =
            (FloatVectorValues)
                values.newInstance(
                    dimension, count, input.slice(name, 0, fp16.length), null, similarity);
        float[][] queries = new float[queryCount][dimension];
        for (float[] q : queries) {
          for (int i = 0; i < dimension; i++) {
            q[i] = in.getFloat();
          }
        }
        cases[c] = new Case(similarity, dimension, stored, queries);
        // The scorer k-NN chose, inside its prefetching wrapper.
        Object prefetching = selectScorer.invoke(null, stored, queries[0], similarity);
        Field delegate = prefetching.getClass().getDeclaredField("delegate");
        delegate.setAccessible(true);
        Class<?> chosen = delegate.get(prefetching).getClass();
        boolean isNative = chosen.getSimpleName().equals("NativeHalfFloatRandomVectorScorer");
        if (isNative != (similarity != VectorSimilarityFunction.DOT_PRODUCT)) {
          throw new IllegalStateException(
              "case " + c + ", " + similarity + ": k-NN chose " + chosen);
        }
      }
      for (int round = 0; round < WARM_UP_ROUNDS; round++) {
        run(cases);
      }
      byte[] expected = run(cases);
      if (!Arrays.equals(expected, run(cases))) {
        throw new IllegalStateException("the scores changed after the warm-up");
      }
      Files.write(Path.of(args[1]), expected);
    } finally {
      // Best effort, so that it never hides what failed above.
      for (IndexInput input : inputs) {
        try {
          input.close();
        } catch (IOException e) {
          System.err.println("closing an input: " + e);
        }
      }
      try (var paths = Files.walk(dir)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
          Files.deleteIfExists(path);
        }
      } catch (IOException | UncheckedIOException e) {
        System.err.println("removing " + dir + ": " + e);
      }
    }
    Field loaded =
        Class.forName("org.opensearch.knn.jni.KNNLibraryLoader").getDeclaredField("loaded");
    loaded.setAccessible(true);
    System.out.println("native libraries: " + (Set<?>) loaded.get(null));
  }

  static byte[] run(Case[] cases) throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    for (Case c : cases) {
      for (float[] query : c.queries) {
        float[] scores = new float[c.stored.size()];
        VectorScorer.Bulk bulk = c.stored.rescorer(query).bulk(null);
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
        ByteBuffer b =
            ByteBuffer.allocate(scores.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float s : scores) {
          b.putFloat(s);
        }
        out.writeBytes(b.array());
      }
    }
    return out.toByteArray();
  }
}
