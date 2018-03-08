package OptimizationTests;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.ContinuousPeaksEvaluationFunction;
import opt.example.CountOnesEvaluationFunction;
import opt.example.FlipFlopEvaluationFunction;
import opt.example.FourPeaksEvaluationFunction;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;

/**
 * A test using the FlipFlop evaluation function
 * @author Rishab Kaup
 * @version 1.0
 */
public class FlipFlopTest {

    public static void run(int N, int iterations) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FlipFlopEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        // try {
        //     PrintStream out = new PrintStream(new FileOutputStream("FlipFlop/RHC_FlipFlop.csv"));
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
        //     PrintStream out = new PrintStream(new FileOutputStream("FlipFlop/SA_FlipFlop.csv"));
        //     System.setOut(out);
        // } catch (Exception e) {
        //     e.printStackTrace();
        // }

        System.out.println("SA");
        for(int i = 0; i < iterations; i++)
        {

            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            long t = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(sa, 200000);
            fit.train();
            System.out.println(ef.value(sa.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));
        }

        // try {
        //     PrintStream out = new PrintStream(new FileOutputStream("FlipFlop/GA_FlipFlop.csv"));
        //     System.setOut(out);
        // } catch (Exception e) {
        //     e.printStackTrace();
        // }

        System.out.println("GA");
        for(int i = 0; i < iterations; i++)
        {

            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
            long t = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            System.out.println(ef.value(ga.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));

        }

        // try {
        //     PrintStream out = new PrintStream(new FileOutputStream("FlipFlop/MIMIC_FlipFlop.csv"));
        //     System.setOut(out);
        // } catch (Exception e) {
        //     e.printStackTrace();
        // }
        System.out.println("MIMIC");
        for(int i = 0; i < iterations; i++)
        {

            MIMIC mimic = new MIMIC(200, 100, pop);
            long t = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(mimic, 1000);
            fit.train();
            System.out.println(ef.value(mimic.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));

        }
    }
}