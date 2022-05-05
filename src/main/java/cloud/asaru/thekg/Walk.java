
package cloud.asaru.thekg;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 *
 * @author Jae
 */
public class Walk {
    private ArrayList<Integer> data = new ArrayList<>();
    public void addLink(int r, int v) {
        data.add(r);
        data.add(v);
    }
    public void reset(int v){
        data.clear();
        data.add(v);
    }
    public List<Integer> getVertices(){
        return IntStream
                .range(0, data.size())
                .filter(i -> i%2==0)
                .map(i -> data.get(i))
                .boxed()
                .collect(Collectors.toList());
    }
    
    public List<Integer> getEdges(){
        return IntStream
                .range(0, data.size())
                .filter(i -> i%2==1)
                .map(i -> data.get(i))
                .boxed()
                .collect(Collectors.toList());
    }
}
