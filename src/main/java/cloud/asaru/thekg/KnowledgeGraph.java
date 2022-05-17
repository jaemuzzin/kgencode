package cloud.asaru.thekg;

import graphs.FlattenedGraph;
import graphs.MappedGraph;
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
public class KnowledgeGraph extends MultiGraph {

    ArrayList<String> entities;
    ArrayList<String> relations;

    public KnowledgeGraph(ArrayList<String> entities, ArrayList<String> relations, SimpleDirectedGraph<Integer, Triple> graph) {
        super(graph);
        this.entities = entities;
        this.relations = relations;
    }

    public KnowledgeGraph() {
        super();
        entities = new ArrayList<>();
        relations = new ArrayList<>();
    }

    public ArrayList<String> getEntities() {
        return entities;
    }

    public ArrayList<String> getRelations() {
        return relations;
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
}
