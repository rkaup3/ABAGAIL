package NeuralNetTest;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.filt.RandomOrderFilter;
import shared.filt.TestTrainSplitFilter;

import java.io.*;
import java.text.DecimalFormat;
import java.util.Scanner;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying if passengers of the Titanic
 * survived
 *
 * @author Rishab Kaup
 * @version 1.0
 */
public class TitanicTestGA {
    private static Instance[] instances = initializeInstances();
    private static DataSet trainingSet;
    private static DataSet testingSet;

    private static int inputLayer = 7, hiddenLayer = 5, outputLayer = 1, trainingIterations = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[1];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[1];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];
    private static String oaName = "GA";
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    private static int[] pop = {50, 100, 150, 200};
    private static int[] mate = {25, 50, 75, 100};
    private static int[] mutate = {0, 25, 50, 75, 100};

    public static void main(String[] args) {
        RandomOrderFilter r = new RandomOrderFilter();
        r.filter(set);

        TestTrainSplitFilter t = new TestTrainSplitFilter(80);

        t.filter(set);

        trainingSet = t.getTrainingSet();
        testingSet = t.getTestingSet();

//        for (int c = 0; c < pop.length; c++) {
//        for (int c = 0; c < mate.length; c++) {
//        for (int c = 0; c < mutate.length; c++) {

            for (int i = 0; i < oa.length; i++) {
                networks[i] = factory.createClassificationNetwork(
                        new int[]{inputLayer, hiddenLayer, outputLayer});
                nnop[i] = new NeuralNetworkOptimizationProblem(trainingSet, networks[i], measure);
            }


//            oa[0] = new StandardGeneticAlgorithm(pop[c], 25, 10, nnop[0]);
//            oa[0] = new StandardGeneticAlgorithm(100, mate[c], 0, nnop[0]);
//            oa[0] = new StandardGeneticAlgorithm(100, 25, mutate[c], nnop[0]);
            oa[0] = new StandardGeneticAlgorithm(100, 25, 25, nnop[0]);

            for (int i = 0; i < oa.length; i++) {
//                try {
////                    PrintStream out = new PrintStream(new FileOutputStream("GA/" + oaName + "pop"
////                            + Integer.toString(pop[c]) + ".csv"));
////                    PrintStream out = new PrintStream(new FileOutputStream("GA/" + oaName + "mate"
////                            + Integer.toString(mate[c]) + ".csv"));
////                    PrintStream out = new PrintStream(new FileOutputStream("GA/" + oaName + "mutate"
////                            + Integer.toString(mutate[c]) + ".csv"));
////                    PrintStream out = new PrintStream(new FileOutputStream("GA/" + oaName + "_final.csv"));
////                    System.setOut(out);
//                } catch (Exception e) {
//                    e.printStackTrace();
//                }

                double[] times = train(oa[i], networks[i]); //trainer.train();

                results += "\nTraining time: " + df.format(times[0])
                        + " seconds\nTesting time: " + df.format(times[1]) + " seconds\n";
            }

            System.out.println(results);
        }


//    }

    private static double[] train(OptimizationAlgorithm oa, BackPropagationNetwork network) {
        System.out.println("Iteration, Train Accuracy, Test Accuracy");
        double trainingTime = 0, testingTime = 0;
        for(int i = 0; i < trainingIterations; i++) {
            double start = System.nanoTime(), end;
            oa.train();
            int trainCorrect = 0;
            for(int j = 0; j < trainingSet.getInstances().length; j++) {
                network.setInputValues(trainingSet.getInstances()[j].getData());
                network.run();

                double predicted = Double.parseDouble(trainingSet.getInstances()[j].getLabel().toString());
                double actual = Double.parseDouble(network.getOutputValues().toString());

                if(Math.abs(predicted - actual) < 0.5)
                {
                    trainCorrect++;
                }
            }
            end = System.nanoTime();
            trainingTime += end - start;

            Instance optimalInstance = oa.getOptimal();
            network.setWeights(optimalInstance.getData());

            start = System.nanoTime();
            int testCorrect = 0;

            for(int j = 0; j < testingSet.getInstances().length; j++) {
                network.setInputValues(testingSet.getInstances()[j].getData());
                network.run();

                double predicted = Double.parseDouble(testingSet.getInstances()[j].getLabel().toString());
                double actual = Double.parseDouble(network.getOutputValues().toString());

                if(Math.abs(predicted - actual) < 0.5)
                {
                    testCorrect++;
                }

            }
            end = System.nanoTime();
            testingTime += end - start;



            System.out.println(Integer.toString(i + 1) + ", " +
                    (100 * ((double)trainCorrect/(double) trainingSet.getInstances().length)) + ", " +
                    (100 * ((double)testCorrect/(double) testingSet.getInstances().length)));


        }
        trainingTime /= Math.pow(10,9);
        testingTime /= Math.pow(10,9);
        return new double[]{trainingTime, testingTime};
    }



    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[1045][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("titanic3_standardized.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[6]; // 7 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 6; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);

            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}