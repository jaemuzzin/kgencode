package cloud.asaru.thekg;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.jgrapht.graph.SimpleDirectedGraph;

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

    @Override
    public int getRelationCount() {
        return relations.size();
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
