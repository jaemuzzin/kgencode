package cloud.asaru.thekg;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.jgrapht.Graph;
import org.jgrapht.Graphs;
import org.jgrapht.graph.AsSubgraph;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleDirectedGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class CompleteGraph {

    ArrayList<String> entities = new ArrayList<>();
    ArrayList<String> relations = new ArrayList<>();
    SimpleDirectedGraph<Integer, Triple> graph = new SimpleDirectedGraph(Triple.class);

    public void build(Reader reader) {
        BufferedReader in = new BufferedReader(reader);
        String line = null;
        try {
            while ((line
                    = in.readLine()) != null) {
                String[] data = line.split("\\t");
                if (data.length != 3) {
                    continue;
                }
                if (!entities.contains(data[0])) {
                    entities.add(data[0]);
                }
                if (!entities.contains(data[2])) {
                    entities.add(data[2]);
                }
                if (!relations.contains(data[1])) {
                    relations.add(data[1]);
                }
                graph.addVertex(entities.indexOf(data[0]));
                graph.addVertex(entities.indexOf(data[2]));
                graph.addEdge(entities.indexOf(data[0]), entities.indexOf(data[2]), new Triple(entities.indexOf(data[0]), relations.indexOf(data[1]), entities.indexOf(data[2])));
            }
        } catch (IOException ex) {
            Logger.getLogger(CompleteGraph.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public SimpleDirectedGraph<Integer, Triple> getCompleteGraph() {
        return graph;
    }

    public Graph<Integer, Triple> subgraph(int u, int v, int kHops) {
        //all edges with v or u as head 
        Set<Triple> filteredEdges = graph.edgeSet()
                .stream().filter(e -> e.getH() == u || e.getH() == v).collect(Collectors.toSet());

        //build khop neighbourhood of u
        Set<Integer> khopU = new HashSet<>();
        khopU.add(u);
        for (int i = 0; i < kHops; i++) {
            khopU.addAll(
                    khopU.stream().flatMap(vi -> Graphs.neighborSetOf(graph, vi).stream())
                            .collect(Collectors.toSet()));
        }
        //of v
        Set<Integer> khopV = new HashSet<>();
        khopV.add(v);
        for (int i = 0; i < kHops; i++) {
            khopV.addAll(
                    khopV.stream().flatMap(vi -> Graphs.neighborSetOf(graph, vi).stream())
                            .collect(Collectors.toSet()));
        }
        //get the intersection
        Set<Integer> intersection = new HashSet<>();
        intersection.addAll(khopU);
        intersection.retainAll(khopV);
        //get edges only in this neighbourhood, add it to list.
        filteredEdges.addAll(graph.edgeSet().stream()
                .filter(e -> intersection.contains(e.getH()) && intersection.contains(e.getT())).collect(Collectors.toSet()));
        return new AsSubgraph<>(graph, graph.vertexSet(), filteredEdges);
    }

    public static INDArray getSpectralNodeCoords(Graph<Integer, Triple> graph, int size, int dims) {

        INDArray r = Nd4j.zeros(size, dims);
        INDArray laplacian = Nd4j.zeros(size, size);
        //diag
        IntStream.range(0, size).forEach(i -> laplacian.putScalar(new int[]{i, i}, graph.edgesOf(i).size()));
        IntStream.range(0, size).forEach(i
                -> IntStream.range(0, size).filter(j -> i!=j).forEach(j
                        -> laplacian.putScalar(new int[]{i, j}, graph.containsEdge(i, j) ? 1 : 0)));
        int feidlerIndex = 0;
        //find the filder
        INDArray[] eig = Eigen.eig(laplacian);
        while (eig[0].getInt(feidlerIndex) <= 0) {
            feidlerIndex++;
        }
        //populate matrix, [node, dimension]
        for (int n = 0; n < size; n++) {
            for (int d = 0; d < dims; d++) {
                r.putScalar(new int[]{n, d}, eig[1].getDouble(feidlerIndex + d, n));
            }
        }
        return r;
    }
}
