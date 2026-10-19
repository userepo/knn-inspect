// Runs k-NN's own FaissMemoryOptimizedSearcher, which searches a Faiss
// graph with Lucene's HnswGraphSearcher through k-NN's FaissHnswGraph, over
// the cases tools/search_oracle.py writes, as the expected results for the
// Rust port. The scores come from the cases, through a scorer that looks
// them up, so only the search is under test. Run with the jars of the
// OpenSearch 3.9.0 image on the class path, in source-file mode:
//
//   java -cp '<lib>/*:<knn plugin>/*' SearchOracle.java <fixtures> <cases> <expected>
//
// <cases>: a little-endian uint32 count, then per case the native file's
// path under <fixtures> (a uint16 length, then UTF-8), the number of
// vectors, the number of queries, and the number of k values (uint32
// each), the k values (uint32 each), the doc id of each vector's ordinal
// (int32 each), and per query the score of every vector (float32 each).
// <expected>: per case, per query, per k, the number of results (uint32),
// each result's doc id (int32) and score (float32), and the visited count
// (int64), little-endian.

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcher;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryHnswIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryIndex;

public class SearchOracle {
  /** Vectors no one reads: the scorer looks the scores up. */
  static final class NoValues extends FloatVectorValues {
    final int size;
    final int dimension;

    NoValues(int size, int dimension) {
      this.size = size;
      this.dimension = dimension;
    }

    @Override
    public float[] vectorValue(int ord) {
      return new float[dimension];
    }

    @Override
    public int dimension() {
      return dimension;
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public NoValues copy() {
      return this;
    }
  }

  /** The storage of a graph without its vectors, which k-NN fills in. */
  static final class NoStorage extends FaissBinaryIndex {
    final int size;
    final int dimension;

    NoStorage(int size, int dimension) {
      super("none");
      this.size = size;
      this.dimension = dimension;
    }

    @Override
    protected void doLoad(IndexInput input) {}

    @Override
    public VectorEncoding getVectorEncoding() {
      return VectorEncoding.FLOAT32;
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput input) {
      return new NoValues(size, dimension);
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput input) {
      throw new UnsupportedOperationException();
    }
  }

  /** Scores looked up in a table, one row per query. */
  static final class Table implements RandomVectorScorer {
    final float[] scores;
    final int[] docs;
    final int n;
    int query;

    Table(float[] scores, int[] docs) {
      this.scores = scores;
      this.docs = docs;
      this.n = docs.length;
    }

    @Override
    public float score(int node) {
      return scores[query * n + node];
    }

    @Override
    public int maxOrd() {
      return n;
    }

    @Override
    public int ordToDoc(int ord) {
      return docs[ord];
    }
  }

  static final class TableScorer implements FlatVectorsScorer {
    final Table table;

    TableScorer(Table table) {
      this.table = table;
    }

    @Override
    public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
        VectorSimilarityFunction similarity, KnnVectorValues values) {
      throw new UnsupportedOperationException();
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarity, KnnVectorValues values, float[] target) {
      return table;
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarity, KnnVectorValues values, byte[] target) {
      throw new UnsupportedOperationException();
    }
  }

  public static void main(String[] args) throws IOException {
    Path fixtures = Path.of(args[0]);
    ByteBuffer in =
        ByteBuffer.wrap(Files.readAllBytes(Path.of(args[1]))).order(ByteOrder.LITTLE_ENDIAN);
    int count = in.getInt();
    ByteArrayOutputStream expected = new ByteArrayOutputStream();
    for (int c = 0; c < count; c++) {
      byte[] name = new byte[Short.toUnsignedInt(in.getShort())];
      in.get(name);
      Path file = fixtures.resolve(new String(name, StandardCharsets.UTF_8));
      int n = in.getInt();
      int queries = in.getInt();
      int[] ks = new int[in.getInt()];
      for (int i = 0; i < ks.length; i++) {
        ks[i] = in.getInt();
      }
      int[] docs = new int[n];
      for (int i = 0; i < n; i++) {
        docs[i] = in.getInt();
      }
      float[] scores = new float[queries * n];
      for (int i = 0; i < scores.length; i++) {
        scores[i] = in.getFloat();
      }

      try (Directory dir = new MMapDirectory(file.getParent());
          IndexInput input = dir.openInput(file.getFileName().toString(), IOContext.DEFAULT)) {
        FaissIndex index = FaissIndex.load(input.clone());
        if (index instanceof FaissIdMapIndex idMap
            && idMap.getNestedIndex() instanceof FaissBinaryHnswIndex graph
            && graph.getStorage() == null) {
          graph.setStorage(new NoStorage(n, index.getDimension()));
        }
        FieldInfo field =
            new FieldInfo(
                "vec",
                0,
                false,
                false,
                false,
                IndexOptions.NONE,
                DocValuesType.NONE,
                DocValuesSkipIndexType.NONE,
                -1,
                Map.of(),
                0,
                0,
                0,
                index.getDimension(),
                VectorEncoding.FLOAT32,
                VectorSimilarityFunction.EUCLIDEAN,
                false,
                false);
        Table table = new Table(scores, docs);
        FaissMemoryOptimizedSearcher searcher =
            new FaissMemoryOptimizedSearcher(input, index, field, new TableScorer(table));
        for (int q = 0; q < queries; q++) {
          table.query = q;
          for (int k : ks) {
            // As k-NN's MemoryOptimizedKNNWeight collects a one-segment
            // search without a filter.
            TopKnnCollector collector =
                new TopKnnCollector(k, Integer.MAX_VALUE, new KnnSearchStrategy.Hnsw(0));
            searcher.search(
                new float[index.getDimension()], collector, AcceptDocs.fromLiveDocs(null, n));
            TopDocs top = collector.topDocs();
            ByteBuffer out =
                ByteBuffer.allocate(4 + 8 * top.scoreDocs.length + 8)
                    .order(ByteOrder.LITTLE_ENDIAN);
            out.putInt(top.scoreDocs.length);
            for (ScoreDoc d : top.scoreDocs) {
              out.putInt(d.doc);
              out.putFloat(d.score);
            }
            out.putLong(top.totalHits.value());
            expected.write(out.array());
          }
        }
      }
    }
    Files.write(Path.of(args[2]), expected.toByteArray());
  }
}
