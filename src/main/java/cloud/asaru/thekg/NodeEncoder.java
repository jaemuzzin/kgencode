
package cloud.asaru.thekg;

import org.jgrapht.Graph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class NodeEncoder {
    private final CharEncoder encoder;
    private int embeddingSize;
    public NodeEncoder(int encoderSize, int embeddingSize, int maxLength) {
        this.embeddingSize = embeddingSize;
        this.encoder = new CharEncoder(encoderSize, embeddingSize, maxLength);
    }

    public int getEmbeddingSize() {
        return embeddingSize;
    }

    public void fit(Graph<Integer, Triple> g, int v) {
        encoder.fit(g.edgesOf(v).stream().mapToInt(t -> t.r).sorted().mapToObj(r -> ""+((char)(1+r))).reduce("", (a,b) -> a+b));
    }

    public String autoencode(Graph<Integer, Triple> g, int v) {
        return encoder.autoencode(g.edgesOf(v).stream().mapToInt(t -> t.r).sorted().mapToObj(r -> ""+((char)(1+r))).reduce("", (a,b) -> a+b));
    }
    
    public INDArray getEmbedding(Graph<Integer, Triple> g, int v){
        return Nd4j.zeros(this.embeddingSize);
    }
    
}
