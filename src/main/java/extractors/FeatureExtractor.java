
package extractors;

import org.nd4j.autodiff.samediff.SDVariable;

/**
 *
 * @author Jae
 */
public interface FeatureExtractor {
    
    //returns and dxN matrix of features for each node, given another DxN matrix where D > d
    public SDVariable extract(SDVariable sd);
    
    public int getDimensions();
}
