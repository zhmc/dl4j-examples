package org.deeplearning4j.examples.dataexamples;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 酒店评论语料IF-IDF方法生成的csv文件，只是使用了单层，就是普通的逻辑回归
 * @author zhminchao@163.com
 * @date 2017年5月12日 上午10:35:52
 */
public class CSVHotelComment {

    private static Logger log = LoggerFactory.getLogger(CSVHotelComment.class);

    public static void main(String[] args) throws  Exception {

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        String delimiter = ",";  //这是csv文件中的分隔符
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("dataSet20170119.csv").getFile()));

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network

        /**
         * 这是标签（类别）所在列的序号
         */
        int labelIndex = 0;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        //类别的总数
        int numClasses = 2;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2

        //一次读取多少条数据。当数据量较大时可以分几次读取，每次读取后就训练。这就是随机梯度下降
        int batchSize = 2000;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        //因为我在这里写的2000是总的数据量大小，一次读完，所以一次next就完了。如果分批的话要几次next
        DataSet allData = iterator.next();
//        allData.
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);  //Use 65% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();
        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        //数据归一化的步骤，非常重要，不做可能会导致梯度中的几个无法收敛
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set


        final int numInputs = 9202;
        int outputNum = 2;
        int iterations = 1000;
        long seed = 142;
        int numEpochs = 50; // number of epochs to perform

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(iterations)
            .activation(Activation.SIGMOID)
            .weightInit(WeightInit.ZERO)  //参数初始化的方法，全部置零
            .learningRate(1e-3)          //经过测试，这里的学习率在1e-3比较合适
            .regularization(true).l2(5e-5) //正则化项，防止参数过多导致过拟合。2阶范数乘一个比率
            .list()

            .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SIGMOID)
                .nIn(numInputs).nOut(outputNum).build())
            .backprop(true).pretrain(false)
            .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));


        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            log.info("Epoch " + i);
            model.fit(trainingData);

            //每个epoch结束后立马观察模型估计的效果，方便及早停止
            Evaluation eval = new Evaluation(numClasses);
            INDArray output = model.output(testData.getFeatureMatrix());
            eval.eval(testData.getLabels(), output);
            log.info(eval.stats());
        }

    }

}

