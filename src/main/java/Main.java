
import cloud.asaru.thekg.CharEncoder;
import java.io.File;
import java.io.IOException;
import java.util.Random;


/**
 *
 * @author Jae
 */
public class Main {
public static final CharEncoder instance = new CharEncoder(10, 20, 40);
    
    
    public static final String datasmallaa = "ABBBCCCDDEEEFGHIJ";
    
    public static final String datasmalljj = "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ";
    public static final String datasmalltbc = "BCBCBCBCBCBCBCBCBCBC";
    public static final String datasmalltde = "DDDDDEEEEEDDDDDEEEEEDDDDDEEEEE";
    public static final String datasmalltde2 = "DDCCCDEEEEEDDDDFFEEEEDDDDDAAEEE";
    public static final String datasmallthi = "HHHHHHHHHHHHHIIIIIIIIII";
    public static void main(String[] args) throws IOException{
       Random rand = new Random(123);
        for(int i=0;i<10000;i++){
            StringBuilder random = new StringBuilder();
            for(int j=0;j<10+rand.nextInt(29);j++) {
                random.append((char)(65+rand.nextInt(10)));
            }
            instance.fit(new String(random));
            if(i%100==0) {
                String result = instance.autoencode(datasmallaa);
                System.out.println(result);
                String result2 = instance.autoencode(datasmalltbc);
                System.out.println(result2);
                String result3 = instance.autoencode(datasmalltde);
                System.out.println(result3);
                String result5 = instance.autoencode(datasmallthi);
                System.out.println(result5);
                String result6 = instance.autoencode(datasmalljj);
                System.out.println(result6);
            }
        }
        instance.rnet.save(new File("model.dat"));
    }
    
}
