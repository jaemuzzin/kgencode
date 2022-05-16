package cloud.asaru.thekg;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
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
import org.jgrapht.graph.SimpleGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class KnowledgeGraph {

    ArrayList<String> entities = new ArrayList<>();
    ArrayList<String> relations = new ArrayList<>();
    SimpleDirectedGraph<Integer, Triple> graph = new SimpleDirectedGraph(Triple.class);

    public KnowledgeGraph() {
    }

    public ArrayList<String> getEntities() {
        return entities;
    }

    public ArrayList<String> getRelations() {
        return relations;
    }

    public SimpleDirectedGraph<Integer, Triple> getGraph() {
        return graph;
    }

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
            Logger.getLogger(KnowledgeGraph.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public SimpleDirectedGraph<Integer, Triple> getCompleteGraph() {
        return graph;
    }

    public SimpleDirectedGraph<Integer, Triple> subgraph(int u, int v, int kHops) {
        SimpleDirectedGraph<Integer, Triple> subgraph = new SimpleDirectedGraph(Triple.class);
        graph.vertexSet().stream().forEach(ve -> subgraph.addVertex(ve));
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
        filteredEdges.forEach(e -> subgraph.addEdge(e.h, e.t, new Triple(e.h, e.r, e.t)));
        return subgraph;
    }

    /*
    [nodes, feature]
    */
    public static INDArray getSpectralNodeCoordsSimpleGraph(SimpleGraph<Integer, DefaultEdge> graph, int numRels, int dims) {

        INDArray r = Nd4j.zeros(graph.vertexSet().size(), dims);
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
        for (int n = 0; n < graph.vertexSet().size(); n++) {
            for (int d = 0; d < dims; d++) {
                //this nodes value in dimension d is the node's value in fidelr vector + d
                r.putScalar(new int[]{n, d}, eig[1].getDouble(feidlerIndex + d, n));
            }
        }
        return r;
    }

    public static INDArray getSpectralNodeCoords(KnowledgeGraph kg, int numNodes, int dims) {
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

    /*
    relationships are negative indexed.
     */
    public static SimpleGraph<Integer, DefaultEdge> toSimpleGraph(SimpleDirectedGraph<Integer, Triple> subgraph, int numNodes, int numRels) {
        Set<Integer> rels = subgraph.edgeSet().stream()
                .map(e -> e.r)
                .distinct()
                .sorted()
                .collect(Collectors.toSet());
        SimpleGraph<Integer, DefaultEdge> rreturn = new SimpleGraph<>(DefaultEdge.class);
        subgraph.vertexSet().stream()
                .forEach(v -> rreturn.addVertex(v));
        rels.stream()
                .forEach(r -> rreturn.addVertex(numNodes+r));
        IntStream oi = IntStream.range(numNodes+numRels, numNodes + numRels+subgraph.edgeSet().size());
        Iterator<Integer> nodeGenerator = oi.iterator();
        subgraph.edgeSet().stream()
                .forEach(e -> {
                    int i = nodeGenerator.next();
                    rreturn.addVertex(i);
                    rreturn.addEdge(e.h, i);
                    rreturn.addEdge(i, e.t);
                    rreturn.addEdge(i, numNodes+e.r);
                });

        return rreturn;
    }
}
