package cloud.asaru.thekg;

import graphs.FlattenedGraph;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.jgrapht.Graph;
import org.jgrapht.Graphs;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleDirectedGraph;
import org.jgrapht.graph.SimpleGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class MultiGraph {

    protected SimpleDirectedGraph<Integer, Triple> graph;

    public MultiGraph(SimpleDirectedGraph<Integer, Triple> graph) {
        this.graph = graph;
    }

    public MultiGraph() {
        graph = new SimpleDirectedGraph(Triple.class);
    }

    public SimpleDirectedGraph<Integer, Triple> getGraph() {
        return graph;
    }

    public void build(Collection<Triple> triples) {
        triples.forEach(tr -> {
            graph.addVertex(tr.h);
            graph.addVertex(tr.t);
            graph.addEdge(tr.h, tr.t, new Triple(tr.h, tr.r, tr.t));
        });
    }

    public SimpleDirectedGraph<Integer, Triple> getCompleteGraph() {
        return graph;
    }

    public int getRelationCount() {
        return (int) graph.edgeSet().stream().mapToInt(e -> e.r).distinct().count();
    }

    public long getNodeCount() {
        return graph.vertexSet().size();
    }

    /*
    * returns a new graph where nodes are 0 to N
     */
    public MultiGraph toSequentialIdGraph() {
        List<Integer> nodes = getSequentialIds();
        SimpleDirectedGraph<Integer, Triple> inner = new SimpleDirectedGraph<>(Triple.class);
        IntStream.range(0, nodes.size()).forEach(i -> inner.addVertex(i));

        graph.edgeSet().stream().forEach(e -> {
            if (nodes.indexOf(e.h) == -1 || nodes.indexOf(e.t) == -1) {
                throw new RuntimeException("nodes not mapped.");
            }
            inner.addEdge(nodes.indexOf(e.h), nodes.indexOf(e.t), new Triple(nodes.indexOf(e.h), e.r, nodes.indexOf(e.t)));
        });
        return new MultiGraph(inner);
    }

    /*
    * returns the origal ids referenced by the sequential ids in toSequentialIdGraph
     */
    public List<Integer> getSequentialIds() {
        List<Integer> nodes = new ArrayList<>();
        //only add nodes with edges
        nodes.addAll(graph.edgeSet().stream().flatMapToInt(e -> Arrays.stream(new int[]{e.h, e.t})).distinct().boxed().collect(Collectors.toList()));
        return nodes;
    }

    public INDArray getMultiRelAdjacencyTensor(int numNodes, int numRels) {
        INDArray r = Nd4j.zeros(numRels, numNodes, numNodes);
        graph.edgeSet().forEach(e -> r.putScalar(new int[]{e.r, e.h, e.t}, 1));
        graph.edgeSet().forEach(e -> r.putScalar(new int[]{e.r, e.t, e.h}, 1));
        return r;
    }

    public int getMinimumHopsBetween(int u, int v) {
        //build khop neighbourhood of u
        Set<Integer> khopU = new HashSet<>();
        khopU.add(u);
        int hops = 0;
        while (!khopU.contains(v) && hops < 10) {
            khopU.addAll(
                    khopU.stream().filter(vi -> graph.containsVertex(vi)).flatMap(vi -> Graphs.neighborSetOf(graph, vi).stream())
                            .collect(Collectors.toSet()));
            hops++;
        }
        return hops;
    }

    public MultiGraph subgraph(int u, int v, int kHops, int maxNodes, int r_ignore) {
        SimpleDirectedGraph<Integer, Triple> subgraph = new SimpleDirectedGraph(Triple.class);
        graph.vertexSet().stream().forEach(ve -> subgraph.addVertex(ve));
//all edges with v or u as head or tail
        Set<Triple> filteredEdges = new HashSet<>();

        //build khop neighbourhood of u
        Set<Integer> intersection = new HashSet<>();
        int hope = 0;
        Set<Integer> khopU = new HashSet<>();
        khopU.add(u);
        Set<Integer> khopV = new HashSet<>();
        khopV.add(v);
        while (intersection.size() < 3 && hope < kHops) {
            khopU.addAll(
                    khopU.stream().filter(f -> Graphs.neighborSetOf(graph, f).size() <= getRelationCount()).flatMap(vi -> Graphs.neighborSetOf(graph, vi).stream().filter(n -> n != v))
                            .collect(Collectors.toSet()));
            //of v
            khopV.addAll(
                    khopV.stream().filter(f -> Graphs.neighborSetOf(graph, f).size() <= getRelationCount()).flatMap(vi -> Graphs.neighborSetOf(graph, vi).stream().filter(n -> n != u))
                            .collect(Collectors.toSet()));
            //get the intersection
            intersection.addAll(khopU);
            intersection.retainAll(khopV);
            hope++;
        }
        intersection.add(v);
        intersection.add(u);
        //get edges only in this neighbourhood, add it to list.
        filteredEdges.addAll(graph.edgeSet().stream()
                .filter(e
                        -> intersection.contains(e.getH()) && intersection.contains(e.getT()))
                .filter(e -> !(e.h == u && e.t == v && e.r == r_ignore))
                .collect(Collectors.toSet()));
        filteredEdges
                .stream()
                .filter(e -> subgraph.edgeSet().stream().flatMapToInt(e2 -> Arrays.stream(new int[]{e2.h, e2.t})).distinct().count() < maxNodes)
                .forEach(e -> subgraph.addEdge(e.h, e.t, new Triple(e.h, e.r, e.t)));

        MultiGraph r = new MultiGraph(subgraph);
        //remove hubs
        /*Set<Integer> hubs = r.graph.vertexSet().stream()
                .filter(f -> f != u && f != v)
                .filter(f -> Graphs.neighborSetOf(graph, f).size() > getRelationCount()).collect(Collectors.toSet());

        hubs.forEach(f -> r.graph.removeVertex(f));*/
        //prune vertices not on path, they will only have 1 connection
        while (r.graph.vertexSet().stream()
                .filter(f -> f != u && f != v)
                .filter(f -> Graphs.neighborSetOf(graph, f).size() == 1).count() > 0) {
            Set<Integer> toremove = r.graph.vertexSet().stream()
                    .filter(f -> f != u && f != v)
                    .filter(f -> Graphs.neighborSetOf(graph, f).size() == 1).collect(Collectors.toSet());

            toremove.forEach(f -> r.graph.removeVertex(f));
        }
        return r;
    }

    /*
    [feature, nodes]
     */
    public INDArray getSpectralNodeCoordsSimpleGraph(int dims, int nodes) {

        INDArray r = Nd4j.zeros(dims, nodes);
        INDArray laplacian = Nd4j.zeros(graph.vertexSet().size(), graph.vertexSet().size());
        //diag
        IntStream.range(0, graph.vertexSet().size()).forEach(i -> laplacian.putScalar(new int[]{i, i}, graph.edgesOf(i).size()));
        IntStream.range(0, graph.vertexSet().size()).forEach(i
                -> IntStream.range(0, graph.vertexSet().size()).filter(j -> i != j).forEach(j
                        -> laplacian.putScalar(new int[]{i, j}, graph.containsEdge(i, j) ? 1 : 0)));
        int feidlerIndex = 0;
        //find the filder
        INDArray[] eig = Eigen.eig(laplacian);
        while (eig[0].getInt(feidlerIndex) <= 0) {
            feidlerIndex++;
        }
        //populate matrix, [node, dimension]
        for (int n = 0; n < nodes; n++) {
            for (int d = 0; d < dims; d++) {
                if (feidlerIndex + d < eig[1].shape()[0] && n < eig[1].shape()[1]) {//this nodes value in dimension d is the node's value in fidelr vector + d
                    r.putScalar(new int[]{d, n}, eig[1].getDouble(feidlerIndex + d, n));
                }
            }
        }
        return r;
    }

    public static INDArray getSpectralNodeCoords(MultiGraph kg, int numNodes, int dims) {
        SimpleDirectedGraph<Integer, Triple> graph = kg.getGraph();
        INDArray r = Nd4j.zeros(numNodes, dims);
        INDArray laplacian = Nd4j.zeros(numNodes, numNodes);
        //diag
        IntStream.range(0, numNodes).forEach(i -> laplacian.putScalar(new int[]{i, i}, graph.edgesOf(i).size()));
        IntStream.range(0, numNodes).forEach(i
                -> IntStream.range(0, numNodes).filter(j -> i != j).forEach(j
                        -> laplacian.putScalar(new int[]{i, j}, graph.containsEdge(i, j) ? 1 : 0)));
        int feidlerIndex = 0;
        //find the filder
        INDArray[] eig = Eigen.eig(laplacian);
        while (eig[0].getInt(feidlerIndex) <= 0) {
            feidlerIndex++;
        }
        //populate matrix, [node, dimension]
        for (int n = 0; n < numNodes; n++) {
            for (int d = 0; d < dims; d++) {
                r.putScalar(new int[]{n, d}, eig[1].getDouble(feidlerIndex + d, n));
            }
        }
        return r;
    }

    public Graph<Integer, DefaultEdge> toSimpleGraph(SimpleDirectedGraph<Integer, Triple> subgraph, int numNodes, int numRels) {

        SimpleGraph<Integer, DefaultEdge> rreturn = new SimpleGraph<>(DefaultEdge.class);
        FlattenedGraph.flatten(subgraph, rreturn);
        return rreturn;
    }
}
