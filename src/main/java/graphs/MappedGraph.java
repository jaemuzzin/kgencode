package graphs;

import cloud.asaru.thekg.Triple;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.IntStream;
import org.jgrapht.Graph;
import org.jgrapht.graph.SimpleDirectedGraph;

/**
 * Creates a mapping from one node id space to another node id space.
 * Accepts an outer graph with nodes from 1 to V
 * and populates another "inner" graph with nodes Rangine from 1 to (E to 2E), where E is the number of edges in outer.
 * 
 * The bijective mapping from outer node id space to inner node id space is stored and accessible for reference
 * @author Jae
 */
public class MappedGraph {

    private int innerNodes;
    private int outerNodes;
    private ArrayList<Integer> map;
    private int mapPointer = 0;
    private SimpleDirectedGraph<Integer, Triple> start;
    private SimpleDirectedGraph<Integer, Triple> inner =new SimpleDirectedGraph<>(Triple.class);

    public MappedGraph(SimpleDirectedGraph<Integer, Triple> outer) {
        map = new ArrayList<>(start.edgeSet().size());
        IntStream.range(0, map.size()).forEach(i -> inner.addVertex(i));
      
        start.edgeSet().stream().forEach(e -> {
            if (map.stream().noneMatch(o -> o.equals(e.t))) {
                map.add(e.t);
            }
            if (map.stream().noneMatch(o -> o.equals(e.h))) {
                map.add(e.h);
            }
            inner.addEdge(map.indexOf(e.h), map.indexOf(e.t), new Triple(map.indexOf(e.h), e.r, map.indexOf(e.t)));
        });
        this.inner = inner;
    }

    //indexes are inner space, values are outer space
    public ArrayList<Integer> getMap() {
        return map;
    }

    public SimpleDirectedGraph<Integer, Triple> getMappedGraph() {
        return inner;
    }
    

}
