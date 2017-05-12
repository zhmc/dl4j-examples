package org.deeplearning4j.examples.dataexamples;

import java.io.File;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * 招投标文本分类demo  
 * <p>这个分类的csv数据是通过IF-IDF方法得到的文本的向量表示</p>
 * <p>我想尝试一下全连接层。结果证明这是失败的尝试，因为我的样本只有10000个，但是加入了全连接层之后
 * 参数有2000*2000.最终的结果是训练集上效果很好，但是测试集的效果很差。
 * 因为样本相比于参数太少，势必会造成过拟合</p>
 * @author zhminchao@163.com
 * @date 2017年3月8日 下午7:23:54
 */
public class ZhaobiaoTrain {

    private static Logger log = LoggerFactory.getLogger(CSVExample.class);

    public static void main(String[] args) throws  Exception {

//    	log.info("is logger work?");
        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        String delimiter = ",";  //这是csv文件中的分隔符
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new File("resources/dataSet_zhaobiao.csv")));

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network

        /**
         * 这是标签（类别）所在列的序号
         */
        int labelIndex = 0;     //类别标签所在列号（从0开始）
        //类别的总数
        int numClasses = 3;     //招投标中有3个类别
        //一次读取多少条数据。当数据量较大时可以分几次读取，每次读取后就训练。这就是随机梯度下降
        int batchSize = 10000;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)


        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        //因为我在这里写的2000是总的数据量大小，一次读完，所以一次next就完了。如果分批的话要几次next
        DataSet allData = iterator.next();
        

        //这一步非常重要，切分训练集和测试集
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
        
        //将归一化的参数保存下来
//        FileUtils.writeLocalObject(normalizer, "model/normalizer0309v2.model");
        
//        List<DataSet> trainList = trainingData.batchBy(50);
        
        final int numInputs = 14788;  //输入的向量维度
        int outputNum = 3;           //输出几个类别
        double learningRate = 0.001;

        int numHiddenNodes = 2000;
        int iterations = 100;     //迭代次数
        long seed = 142;        //随机种子，为了复现结果
        int numEpochs = 30; // number of epochs to perform

        log.info("Build model....");
        /*
        //多层的配置对象
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)  //最优化方法：随机梯度下降
            .iterations(iterations)
            .activation(Activation.SIGMOID)
            .weightInit(WeightInit.ZERO)  //参数初始化的方法，全部置零
            .learningRate(1e-3)          //经过测试，这里的学习率在1e-3比较合适
//            .regularization(true).l2(5e-5) //正则化项，防止参数过多导致过拟合。2阶范数乘一个比率
            .list()
//            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(500)
//                .build())
//            .layer(1, new DenseLayer.Builder().nIn(500).nOut(500)
//                .build())
            .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SIGMOID)
                .nIn(numInputs).nOut(outputNum).build())
            .backprop(true).pretrain(false)
            .build();
        */
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .regularization(true).l2(5e-5)
//                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                        .nIn(numHiddenNodes).nOut(outputNum).build())
                .pretrain(false).backprop(true).build();
        

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            log.info("Epoch " + i);
//            for (int j = 0; j < trainList.size(); j++) {
            	model.fit(trainingData);
//			}
            

            //每个epoch结束后观察一下模型估计的效果，方便及早停止
            Evaluation eval = new Evaluation(numClasses);
            INDArray output = model.output(trainingData.getFeatureMatrix());
            eval.eval(trainingData.getLabels(), output);
            log.info("train error:------------");
            log.info(eval.stats());
            
            Evaluation eval2 = new Evaluation(numClasses);
            INDArray output2 = model.output(testData.getFeatureMatrix());
            eval2.eval(testData.getLabels(), output2);
            log.info("test error:------------");
            log.info(eval2.stats());
            
        }
        
      boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
      ModelSerializer.writeModel(model, "model/zhaobiao_network_0309v2.zip", saveUpdater);

    }

}

