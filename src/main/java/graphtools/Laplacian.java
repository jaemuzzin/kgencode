
package graphtools;

import java.util.stream.IntStream;
import org.graalvm.compiler.nodes.NodeView;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class Laplacian {
    public INDArray getLaplacian(SimpleGraph<Integer,DefaultEdge> graph) {
        int numNodes = graph.vertexSet().size();
        INDArray laplacian = Nd4j.zeros(numNodes, numNodes);
        //diag
        IntStream.range(0, numNodes).forEach(i -> laplacian.putScalar(new int[]{i, i}, graph.edgesOf(i).size()));
        IntStream.range(0, numNodes).forEach(i
                -> IntStream.range(0, numNodes).filter(j -> i != j).forEach(j
                        -> laplacian.putScalar(new int[]{i, j}, graph.containsEdge(i, j) ? 1 : 0)));
        return laplacian;
    }
}
