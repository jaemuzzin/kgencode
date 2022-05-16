package graphs;

import cloud.asaru.thekg.Triple;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.IntStream;
import org.jgrapht.Graph;

/**
 *
 * @author Jae
 */
public class MappedGraph {

    private int innerNodes;
    private int outerNodes;
    private ArrayList<Integer> map;
    private int mapPointer = 0;
    private Graph<Integer, Triple> start;
    private Graph<Integer, Triple> inner;

    public MappedGraph(Graph<Integer, Triple> outer, Graph<Integer, Triple> inner) {
        map = new ArrayList<>(start.edgeSet().size());
        start.vertexSet().stream().forEach(v -> {
            if (map.stream().noneMatch(o -> o.equals(v))) {
                map.add(v);
            }
        });
        IntStream.range(0, map.size()).forEach(i -> inner.addVertex(i));
      
        start.edgeSet().stream().forEach(e -> {
            inner.addEdge(map.indexOf(e.h), map.indexOf(e.t), new Triple(map.indexOf(e.h), e.r, map.indexOf(e.t)));
        });
        this.inner = inner;
    }

    //indexes are inner space, values are outer space
    public ArrayList<Integer> getMap() {
        return map;
    }

    public Graph<Integer, Triple> getMappedGraph() {
        return inner;
    }
    

}
