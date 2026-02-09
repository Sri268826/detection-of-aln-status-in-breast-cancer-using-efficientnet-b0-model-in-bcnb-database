import java.util.ArrayList;
import java.util.List;

public class lc219 {
    public static boolean res(int[] nums,int k){
        List<Integer> se= new ArrayList<>();
        boolean bool=false;
        for(int i =0;i<nums.length;i++){
            if(se.contains(nums[i])){
                int indi=se.lastIndexOf(nums[i]);
                int j=i;
                //System.out.println(indi+ " " +j);
                if(indi!=j){
                //System.out.println("enter" +indi+" "+j);
                bool= (Math.abs(indi - j) <= k);
                //System.out.println(bool);
                }
                if(!bool) se.add(nums[i]);

            }else{
             se.add(nums[i]);
            }
        }
        return bool;
    }
    public static void main(String[] args) {
        int[] nums={1,0,1,1};
        int k=1;
        System.out.println(res(nums,k));
    }
}
