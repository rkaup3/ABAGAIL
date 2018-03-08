package NeuralNetTest;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import shared.filt.RandomOrderFilter;
import shared.filt.TestTrainSplitFilter;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying if passengers of the Titanic
 * survived
 *
 * @author Rishab Kaup
 * @version 1.0
 */
public class TitanicTestRHC {
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
    private static String oaName = "RHC";
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        RandomOrderFilter r = new RandomOrderFilter();
        r.filter(set);

        TestTrainSplitFilter t = new TestTrainSplitFilter(80);

        t.filter(set);

        trainingSet = t.getTrainingSet();
        testingSet = t.getTestingSet();


        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(trainingSet, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
//        oa[1] = new SimulatedAnnealing(1E11, .5, nnop[1]);
//        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
//            try {
//                PrintStream out = new PrintStream(new FileOutputStream("RHC/" + oaName + "_final.csv"));
//                System.setOut(out);
//            } catch (Exception e) {
//                e.printStackTrace();
//            }


            double[] times = train(oa[i], networks[i]); //trainer.train();


            results +=  "\nTraining time: " + df.format(times[0])
                    + " seconds\nTesting time: " + df.format(times[1]) + " seconds\n";
        }

        System.out.println(results);
    }

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