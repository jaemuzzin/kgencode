
package cloud.asaru.thekg;

/**
 *
 * @author Jae
 */
public abstract class NodeIndexToSequenceMapper {
    public abstract int  getSequenceSize();
    public abstract void setSequenceSize(int n);
    public abstract int mapToSeq(int vertexId);
    public abstract int mapToVertextId(int seq);
}
