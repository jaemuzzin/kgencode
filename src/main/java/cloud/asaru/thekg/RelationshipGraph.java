package cloud.asaru.thekg;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.jgrapht.Graph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class RelationshipGraph {

    int size;
    INDArray laplacian;

    public RelationshipGraph(int size) {
        laplacian = Nd4j.zeros(size, size);
        this.size = size;
    }

    public void build(Graph<Integer, Triple> g) {
        build(g.edgeSet());
    }
    public void build(Collection<Triple> triples) {
        //group by rel type
        Map<Integer, List<Triple>> byR = new HashMap<>();
        triples.forEach(t -> byR.put(t.getR(), new ArrayList<>()));
        triples.forEach(t -> byR.get(t.getR()).add(t));

        //set diagonals
        byR.keySet().forEach(r -> laplacian.putScalar(new int[]{r, r}, byR.get(r).size()));
        byR.keySet().forEach(r -> {
            byR.get(r).forEach(t -> {
                //find s with most matching h and t
                HashMap<Integer, Integer> totals = new HashMap<>();
                triples.forEach(tr -> totals.put(tr.getR(), 0));
                //add to totals
                triples.stream()
                        .peek(tr -> totals.put(tr.getR(), totals.get(tr.getR()) + (t.h == tr.h ? 1 : 0)))
                        .forEach(tr -> totals.put(tr.getR(), totals.get(tr.getR()) + (t.t == tr.t ? 1 : 0)));
                int s = totals.entrySet().stream().max((e1, e2) -> e1.getValue().compareTo(e2.getValue()))
                        .get().getKey();
                laplacian.putScalar(new int[]{t.r, s}, laplacian.getInt(new int[]{t.r, s}) + 1);

            });
        });

    }

    public INDArray getLaplacian() {
        return laplacian;
    }

    public INDArray[] getEigenDecomposition() {
        INDArray[] eig = Eigen.eig(laplacian);
        return eig;
    }

    public INDArray getSpectralCoordinates(int dimensions) {
        INDArray r = Nd4j.zeros(size, dimensions);
        int feidlerIndex = 0;
        //find the filder
        INDArray[] eig = getEigenDecomposition();
        while (eig[0].getInt(feidlerIndex) <= 0) {
            feidlerIndex++;
        }
        //populate matrix, [relationship, dimension]
        for (int rel = 0; rel < size; rel++) {
            for (int d = 0; d < dimensions; d++) {
                r.putScalar(new int[]{rel, d}, eig[1].getDouble(feidlerIndex + d, rel));
            }
        }
        return r;
    }
}
