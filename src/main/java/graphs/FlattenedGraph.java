
package graphs;

import cloud.asaru.thekg.Triple;
import java.util.Iterator;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleGraph;

/**
 * Flattens a multirelation graph to simple undirected graph
 * @author Admin
 */
public class FlattenedGraph {
    public static void flatten(Graph<Integer, Triple> multiRel, Graph<Integer, DefaultEdge> output) {
         Set<Integer> rels = multiRel.edgeSet().stream()
                .map(e -> e.r)
                .distinct()
                .sorted()
                .collect(Collectors.toSet());
        multiRel.vertexSet().stream()
                .forEach(v -> output.addVertex(v));
        rels.stream()
                .forEach(r -> output.addVertex(multiRel.vertexSet().size()+r));
        IntStream oi = IntStream.range(multiRel.vertexSet().size()+rels.size(), multiRel.vertexSet().size()+rels.size()+multiRel.edgeSet().size());
        Iterator<Integer> nodeGenerator = oi.iterator();
        multiRel.edgeSet().stream()
                .forEach(e -> {
                    int i = nodeGenerator.next();
                    output.addVertex(i);
                    output.addEdge(e.h, i);
                    output.addEdge(i, e.t);
                    output.addEdge(i, multiRel.vertexSet().size()+e.r);
                });

    }
}
