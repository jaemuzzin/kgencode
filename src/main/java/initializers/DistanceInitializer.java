/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package initializers;

import cloud.asaru.thekg.MultiGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Admin
 */
public class DistanceInitializer implements NodeInitializer {

    private int[][] distCache = new int[1000][1000];

    public DistanceInitializer() {
        for (int i = 0; i < distCache.length; i++) {
            for (int j = 0; j < distCache[i].length; j++) {
                distCache[i][j] = -1;
            }
        }
    }

    @Override
    public INDArray embed(MultiGraph graph, int dims, int nodes, int h, int t) {
        if (dims != 2) {
            throw new IllegalArgumentException("dims must be 2 for distance initializer");
        }
        INDArray r = Nd4j.zeros(dims, nodes);
        double base_dist = (double) getDistance(graph, h, t);
        if (base_dist > 0) {
            for (int n = 0; n < Math.min(nodes, graph.getNodeCount()); n++) {
                double f1 = (base_dist - getDistance(graph, h, n)) / base_dist;
                double f2 = (base_dist - getDistance(graph, t, n)) / base_dist;
                r.putScalar(new int[]{0, n}, f1);
                r.putScalar(new int[]{1, n}, f2);
            }
        }
        return r;
    }

    private int getDistance(MultiGraph graph, int u, int v) {
        if (distCache[u][v] == -1) {
            distCache[u][v] = graph.getMinimumHopsBetween(u, v);
            distCache[v][u] = distCache[u][v];
        }
        return distCache[u][v];
    }

}
