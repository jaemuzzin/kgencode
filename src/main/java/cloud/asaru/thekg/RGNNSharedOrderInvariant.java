package cloud.asaru.thekg;

import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class RGNNSharedOrderInvariant extends RGNNShared implements RelGNN{

    public RGNNSharedOrderInvariant() {
    }

    @Override
    public RelGNN build(INDArray relationShipAdjTensor, int numNodes, int layers, boolean learnable, boolean sigmoid) {
        return new RGNNSharedOrderInvariant(relationShipAdjTensor, numNodes, layers, sigmoid);
    }
    private RGNNSharedOrderInvariant(INDArray relationShipAdjTensor, int numNodes, int layers, boolean sigmoid) {
        super(relationShipAdjTensor, numNodes, layers, sigmoid);
    }

    /*
    * input is shape [feature index, node index], output is same
     */
    @Override
    public INDArray output(INDArray input) {
        INDArray r = Nd4j.zeros(input.shape());
        for (int i = 0; i < input.shape()[0]; i++) {
            INDArray inp = input.tensorAlongDimension(i, 1, 2);
            //reorder the nodes here:
            sd.getVariable("X").setArray(inp);
            //reorder them back here:
            r.putColumn(i, identity.eval());
        }
        return r;
    }

    /*
    * input is shape [feature index, node index]
     */
    @Override
    public void fit(INDArray input, INDArray output) {

        for (int i = 0; i < input.shape()[0]; i++) {
            INDArray inp = input.tensorAlongDimension(i, 1, 2);
            INDArray outp = output.tensorAlongDimension(i, 1, 2);
            //reorder the nodes here:
            sd.getVariable("X").setArray(inp);
            DataSet ds = new DataSet(input, outp);
            sd.fit(ds, new ScoreListener(200));
        }
    }

}
