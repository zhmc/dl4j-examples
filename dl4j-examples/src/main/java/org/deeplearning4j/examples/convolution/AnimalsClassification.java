package org.deeplearning4j.examples.convolution;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Animal Classification
 *
 * Example classification of photos from 4 different animals (bear, duck, deer, turtle).
 *
 * References:
 *  - U.S. Fish and Wildlife Service (animal sample dataset): http://digitalmedia.fws.gov/cdm/
 *  - Tiny ImageNet Classification with CNN: http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf
 *
 * CHALLENGE: Current setup gets low score results. Can you improve the scores? Some approaches:
 *  - Add additional images to the dataset
 *  - Apply more transforms to dataset
 *  - Increase epochs
 *  - Try different model configurations
 *  - Tune by adjusting learning rate, updaters, activation & loss functions, regularization, ...
 */

public class AnimalsClassification {
    protected static final Logger log = LoggerFactory.getLogger(AnimalsClassification.class);
	//照片是100*100
    protected static int height = 100;
    protected static int width = 100;
	//过滤器数量，就是输入层和几个过滤器连接，每个过滤器都按不同规则对输入层进行处理
    protected static int channels = 3;
    protected static int numExamples = 80;
    protected static int numLabels = 4;
	//每次处理20个样本，这80个样本分4批训练完，参数会更新4次，80个样本训练完了才是一步训练
    protected static int batchSize = 20;
    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int listenerFreq = 1;//参数更新一次，就打印一次score
	//每步训练的迭代次数，正常来讲4批是一次，但是我们可以增加迭代次数让每步训练迭代更多次
    protected static int iterations = 1;
    protected static int epochs = 50;
    protected static double splitTrainTest = 0.8;//80%训练，20%测试
	protected static int nCores = 2; //装载数据的队列数
    protected static boolean save = false; //是否存储模型

    protected static String modelType = "AlexNet"; // LeNet, AlexNet or Custom but you need to fill it out

    public void run(String[] args) throws Exception {

        log.info("Load data....");
        /**cd
         * Data Setup -> organize and limit data file paths:
         *  - mainPath = path to image files  //图片路径
         *  - fileSplit = define basic dataset split with limits on format //定义数据划分
         *  - pathFilter = define additional file load filter to limit size and balance batch content//定义额外的文件加载过滤器用来限制大小平衡批内容
         **/
        
        //按文件名产生标签0,1,2,3
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        //图片主路径
        File mainPath = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/animals/");
        //把所有图片弄成一个经过shuffle的数组
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        //平衡每个batch中label的数量
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);

        /**
         * Data Setup -> train test split //划分训练和测试集
         *  - inputSplit = define train and test split
         **/
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        /**
         * Data Setup -> transformation //构建图片转换的实例，不翻转或者只进行水平垂直翻转
         *  - Transform = how to tranform images and generate large dataset to train on
         **/
        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        //这个实例可以进行翻转，最大翻转为42
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
//        ImageTransform colorTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
        //这就符合我们说的3个channel了
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, warpTransform, flipTransform2});

        /**
         * Data Setup -> normalization
         *  - how to normalize images and generate large dataset to train on
         **/
        //把像素规范化到0,1区间
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        log.info("Build model....");

        // Uncomment below to try AlexNet. Note change height and width to at least 100
//        MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();

        MultiLayerNetwork network;
        switch (modelType) {
            case "LeNet":
                network = lenetModel();
                break;
            case "AlexNet":
                network = alexnetModel();
                break;
            case "custom":
                network = customModel();
                break;
            default:
                throw new InvalidInputTypeException("Incorrect model provided.");
        }
        network.init();
        //设置监听器，参数更新一次就打印一次score
        network.setListeners(new ScoreIterationListener(listenerFreq));

        /**
         * Data Setup -> define how to load data into net:
         * //定义如何把数据载入网络
         *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
         *  //reader装载并转换图片数据，并传入数组inputSplit完成初始化
         *  - dataIter = a generator that only loads one batch at a time into memory to save memory
         *  //数据迭代器，每次只载入一批数据到内存
         *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
         *  //训练迭代器，使用多步迭代器保证模型每步迭代都能使用所有数据 
         **/
        //构建图片读取器
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        //数据迭代器
        DataSetIterator dataIter;
        //训练迭代器
        MultipleEpochsIterator trainIter;


        log.info("Train model....");
        // Train without transformations //训练不翻转的数据，目的是跑一遍初始化参数
        recordReader.initialize(trainData, null);//初始化读取器
        //构造数据迭代器，传入读取器，批大小，label索引，label数量
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        //训练归一器
        scaler.fit(dataIter);
        //对数据进行规范化
        dataIter.setPreProcessor(scaler);
        //构建训练迭代器传入步数，数据迭代器，队列数
        trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores);
        network.fit(trainIter);

        // Train with transformations 训练翻转的数据，有了初始化参数再上各种翻转数据
        for (ImageTransform transform : transforms) { //3组翻转过滤器，代码和之前一样
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            recordReader.initialize(trainData, transform);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);
            trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores);
            network.fit(trainIter);
        }

        log.info("Evaluate model....");
        //初始化测试数据
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        //统计信息是分批统计的，这个要注意下
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        //评估测试数据
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(true));

        // Example on how to get predict results with trained model
        // 如何获取预测结果
        //重置数据迭代器
        dataIter.reset();
        //重置后再next获取的是一条数据
        DataSet testDataSet = dataIter.next();
        //获取label，索引是0
        String expectedResult = testDataSet.getLabelName(0);
        //预测这个数据，返回的是对个各类的概率，所以是列表
        List<String> predict = network.predict(testDataSet);
        //它会把概率最大的放到0的位置，所以get(0)就得到预测值了
        String modelResult = predict.get(0);
        System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelResult + "\n\n");

        if (save) {
            log.info("Save model....");
            //基础路径
            String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
            //保存网络，参数和更新器
            ModelSerializer.writeModel(network, basePath + "model.bin", true);
        }
        log.info("****************Example finished********************");
    }
    
	//卷积输入层，参数包括名字，过滤器数量，输出节点数，卷积核大小，步副大小，补充边框大小，偏差
    /**
     * 卷积输入层
     * @param name 名字
     * @param in 过滤器数量
     * @param out 输出节点数
     * @param kernel 卷积核大小
     * @param stride 步副大小
     * @param pad 补充边框大小
     * @param bias 偏差
     * @return
     */
    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
    	return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }
    
	//3*3的卷积层，卷积核大小3*3，步副大小1*1，补充边框1*1
    private ConvolutionLayer conv3x3(String name, int out, double bias) {
    	return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

	//5*5的卷积层，卷积核大小5*5
    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

	//子采样层，本例中卷积核大小是2*2，步副2*2
    private SubsamplingLayer maxPool(String name,  int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

	//全连接层，本例中输出4096个节点，偏差为1，随机丢弃比例50%，参数服从均值为0，方差为0.005的高斯分布
    private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }

    public MultiLayerNetwork lenetModel() {
        /**
         * Revisde Lenet Model approach developed by ramgo2 achieves slightly above random
         * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
         **/
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .regularization(false).l2(0.005) // tried 0.0001, 0.0005
            .activation(Activation.RELU)
            .learningRate(0.0001) // tried 0.00001, 0.00005, 0.000001
            //参数服从均值为0，方差为2.0/(fanIn + fanOut)的高斯分布，fanIn是上一层节点数，fanOut是当前层节点数
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            //采用可变学习率，动量衰减参数为0.9的参数优化方法
            .updater(Updater.RMSPROP).momentum(0.9)
            .list()
            .layer(0, convInit("cnn1", channels, 50 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
            .layer(1, maxPool("maxpool1", new int[]{2,2}))
            .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
            .layer(3, maxPool("maxool2", new int[]{2,2}))
            .layer(4, new DenseLayer.Builder().nOut(500).build())
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(numLabels)
                .activation(Activation.SOFTMAX)
                .build())
            .backprop(true).pretrain(false) //使用后向反馈，不使用预训练
            //输入的高，宽，过滤器数量
            .setInputType(InputType.convolutional(height, width, channels))
            .build();

        return new MultiLayerNetwork(conf);

    }

    public MultiLayerNetwork alexnetModel() {
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/

        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            //根据给定的分布采样参数
            .weightInit(WeightInit.DISTRIBUTION)
            //均值为0，方差为0.01的正态分布
            .dist(new NormalDistribution(0.0, 0.01))
            .activation(Activation.RELU)
            .updater(Updater.NESTEROVS) //采用梯度修正的参数优化方法
            .iterations(iterations)
            //采用除以梯度2范数来规范化梯度防止梯度消失或突变
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(1e-2)  
            .biasLearningRate(1e-2*2)  //偏差学习率
            .learningRateDecayPolicy(LearningRatePolicy.Step) //按步按批衰减学习率
            .lrPolicyDecayRate(0.1)  //设置衰减率
            .lrPolicySteps(100000)   //设置衰减步
            .regularization(true)
            .l2(5 * 1e-4)
            .momentum(0.9)
            .miniBatch(false) //不限制最小批
            .list()
            //13层的网络,第1,3层构建了alexnet计算层，目的是对当前输出的结果做平滑处理，参数有相邻核映射数n=5,规范化常亮k=2,指数常量beta=0.75，系数常量alpha=1e-4
            .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
            .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
            .layer(2, maxPool("maxpool1", new int[]{3,3}))
            .layer(3, conv5x5("cnn2", 256, new int[] {1,1}, new int[] {2,2}, nonZeroBias))
            .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
            .layer(5, maxPool("maxpool2", new int[]{3,3}))
            .layer(6,conv3x3("cnn3", 384, 0))
            .layer(7,conv3x3("cnn4", 384, nonZeroBias))
            .layer(8,conv3x3("cnn5", 256, nonZeroBias))
            .layer(9, maxPool("maxpool3", new int[]{3,3}))
            .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
            .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
            .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("output")
                .nOut(numLabels)
                .activation(Activation.SOFTMAX)
                .build())
            .backprop(true)
            .pretrain(false)
            .setInputType(InputType.convolutional(height, width, channels))
            .build();

        return new MultiLayerNetwork(conf);

    }

    public static MultiLayerNetwork customModel() {
        /**
         * Use this method to build your own custom model.
         **/
        return null;
    }

    public static void main(String[] args) throws Exception {
        new AnimalsClassification().run(args);
    }

}
