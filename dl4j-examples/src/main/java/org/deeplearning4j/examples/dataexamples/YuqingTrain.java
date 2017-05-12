package org.deeplearning4j.examples.dataexamples;

import java.io.File;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.miscUtil.FileUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * 舆情信息文本分类
 * <p>虽然这个同招标文档分类的程序大体上都是差不多的，都是直接读取IF-IDF生成的文本向量表示CSV。
 * 但是这次由于总词数有5w，文档约54,716个，最终生成的csv文件有几十GB。所有这个程序里面展示了
 * 如何批量读取训练集数据并训练，同时如何批量读取测试集并获取预测评价</p>
 * <p>还展示了数据归一化的步骤，这个对产生良好的模型效果是很重要的</p>
 * <p>模型持久化，归一器持久化</p>
 * @author zhminchao@163.com
 * @date 2017年3月13日 下午4:23:54
 */
public class YuqingTrain {

    private static Logger log =  LoggerFactory.getLogger(YuqingTrain.class);

    public static void main(String[] args) throws  Exception {

//    	log.info("is logger work?");
        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        String delimiter = ",";  //这是csv文件中的分隔符
//        RecordReader trainRecordReader = new CSVRecordReader(numLinesToSkip,delimiter);
//        trainRecordReader.initialize(new FileSplit(new File("resources/train0317.csv")));
//
//        RecordReader testRecordReader = new CSVRecordReader(numLinesToSkip,delimiter);
//        testRecordReader.initialize(new FileSplit(new File("resources/test0317.csv")));

        
        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network

        /**
         * 这是标签（类别）所在列的序号
         */
        int labelIndex = 0;     //类别标签所在列号（从0开始）
        //类别的总数
        int numClasses = 22;     //招投标中有3个类别
        //一次读取多少条数据。当数据量较大时可以分几次读取，每次读取后就训练。这就是随机梯度下降
        int batchSize = 1000;    //一次读取1000条记录

//        训练集迭代读取器
//        DataSetIterator trainIterator = new RecordReaderDataSetIterator(trainRecordReader,batchSize,labelIndex,numClasses);
//        测试集迭代读取器
//        DataSetIterator testIterator = new RecordReaderDataSetIterator(testRecordReader,batchSize,labelIndex,numClasses);
       
        
        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        //数据归一化的步骤，非常重要，不做可能会导致梯度中的几个无法收敛
        DataNormalization normalizer = new NormalizerStandardize();
//        log.info("数据开始归一化");
//        normalizer.fit(trainIterator);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
//        log.info("数据结束归一化");
//        //将归一化的参数保存下来
//        FileUtils.writeLocalObject(normalizer, "model/normalizer0317.model");
        log.info("加载normalizer");
        normalizer=FileUtils.readLocalObject("model/normalizer0317.model", NormalizerStandardize.class);
        
        final int numInputs = 40000;  //输入的向量维度
        int outputNum = numClasses;           //输出几个类别
        int iterations = 1;     //迭代次数
        long seed = 42;        //随机种子，为了复现结果
        int numEpochs = 30; // number of epochs to perform

        log.info("Build model....");

        //这个地方的模型其实就是个单纯的逻辑回归分类器，用了softmax做分类函数。并不是多层网络
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)  //最优化方法：随机梯度下降
            .iterations(iterations)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.ZERO)  //参数初始化的方法，全部置零
            .learningRate(5e-3)          //经过测试，这里的学习率在1e-3比较合适
            .regularization(true).l2(10e-5) //正则化项，防止参数过多导致过拟合。2阶范数乘一个比率
            .list()
            .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(numInputs).nOut(outputNum).build())
            .backprop(true).pretrain(false)
            .build();


        //train the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));
        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            log.info("Epoch " + i);
            
            //每个epoch从头开始读取数据，训练模型
            
            RecordReader trainRecordReader2 = new CSVRecordReader(numLinesToSkip,delimiter);
            trainRecordReader2.initialize(new FileSplit(new File("resources/train0317.csv")));
            DataSetIterator  trainIterator2 = new RecordReaderDataSetIterator(trainRecordReader2,batchSize,labelIndex,numClasses);

            while (trainIterator2.hasNext()) {
	            DataSet oneBatchData = trainIterator2.next();
	            normalizer.transform(oneBatchData);
	            model.fit(oneBatchData);
            }
            
            //每个epoch结束后观察一下模型估计的效果，方便及早停止
            log.info("开始评估模型-----------");
            RecordReader testRecordReader2 = new CSVRecordReader(numLinesToSkip,delimiter);
            testRecordReader2.initialize(new FileSplit(new File("resources/test0317.csv")));

            DataSetIterator testIterator = new RecordReaderDataSetIterator(testRecordReader2,batchSize,labelIndex,numClasses);
            Evaluation eval = new Evaluation(numClasses);
            while (testIterator.hasNext()) {
            	DataSet oneBatchTestData = testIterator.next();
	            normalizer.transform(oneBatchTestData);
	            INDArray output = model.output(oneBatchTestData.getFeatureMatrix());
	            eval.eval(oneBatchTestData.getLabels(), output);
			}
            
            log.info(eval.stats());
            
            boolean saveUpdater = true;        //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
            ModelSerializer.writeModel(model, "model/network_0317.zip", saveUpdater);
        }
      
    }

}

