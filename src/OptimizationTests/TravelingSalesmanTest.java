package OptimizationTests;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.*;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;

/**
 * A test using the 4 Peaks evaluation function
 * @author Rishab Kaup
 * @version 1.0
 */
public class TravelingSalesmanTest {

    public static void run(int N, int iterations) {

        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        // try {
        //     PrintStream out = new PrintStream(new FileOutputStream("TravelingSalesman/RHC_TravelingSalesman.csv"));
        //     System.setOut(out);
        // } catch (Exception e) {
        //     e.printStackTrace();
        // }
        System.out.println("RHC");
        for(int i = 0; i < iterations; i++)
        {

            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            long t = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
            fit.train();
            System.out.println(ef.value(rhc.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));
        }

        // try {
        //     PrintStream out = new PrintStream(new FileOutputStream("TravelingSalesman/SA_TravelingSalesman.csv"));
        //     System.setOut(out);
        // } catch (Exception e) {
        //     e.printStackTrace();
        // }
        System.out.println("SA");
        for(int i = 0; i < iterations; i++)
        {

            SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
            long t = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(sa, 200000);
            fit.train();
            System.out.println(ef.value(sa.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));
        }

        // try {
        //     PrintStream out = new PrintStream(new FileOutputStream("TravelingSalesman/GA_TravelingSalesman.csv"));
        //     System.setOut(out);
        // } catch (Exception e) {
        //     e.printStackTrace();
        // }
        System.out.println("GA");
        for(int i = 0; i < iterations; i++)
        {

            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 10, gap);
            long t = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            System.out.println(ef.value(ga.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));
        }



        // for mimic we use a sort encoding
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges);

        // try {
        //     PrintStream out = new PrintStream(new FileOutputStream("TravelingSalesman/MIMIC_TravelingSalesman.csv"));
        //     System.setOut(out);
        // } catch (Exception e) {
        //     e.printStackTrace();
        // }
        System.out.println("MIMIC");
        for(int i = 0; i < iterations; i++)
        {

            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
            MIMIC mimic = new MIMIC(200, 60, pop);
            long t = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(mimic, 1000);
            fit.train();
            System.out.println(ef.value(mimic.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));
        }
    }
}
