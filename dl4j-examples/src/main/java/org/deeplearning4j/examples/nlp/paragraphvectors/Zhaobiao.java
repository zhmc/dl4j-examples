package org.deeplearning4j.examples.nlp.paragraphvectors;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.examples.miscUtil.GBKFileLabelAwareIterator;
import org.deeplearning4j.examples.miscUtil.HanlpTokenizerFactory;
import org.deeplearning4j.examples.nlp.paragraphvectors.tools.LabelSeeker;
import org.deeplearning4j.examples.nlp.paragraphvectors.tools.MeansBuilder;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.util.List;

/**
 * 这个招投标用paragraphVectors分类的效果很差。准确只有29%，远不如酒店评论语料集上的效果（78%）
 * 我想了一下，将原因批注在这里
 * <p>因为在这里paragraphVectors的得来是将一篇文章的所有单词的词向量求平均得来的。
 * 分类是依靠对paragraphVectors得到的带标签向量，做knn(k=1)，得到预测分类。
 * 所以这个模型的本质效果还是看word2vec。但是word2vec不是类似knn或决策树这种可伸缩模型（模型的复杂度能
 * 自动随着样本数量变化）,它的复杂度是固有的。在这个招投标语料集样本数量只有700多的前提下，是无法训练好100维的词向量。
 * 因为无论是得到的词典大小（测试集中估计有太多训练集未涵盖的新词），还是训练程度（对词和词之间关系的捕捉）都远远不达标。
 * 而酒店评论语料集的训练集有8000个。
 * </p>
 */
public class Zhaobiao {

	ParagraphVectors paragraphVectors; //声明ParagraphVectors类
	
	//声明要实现的迭代器接口，用来识别句子或文档及标签，这里假定所有的文档已变成字符串或词表的形式
	LabelAwareIterator iterator;       
	
	TokenizerFactory tokenizerFactory; //声明字符串分割器    

	private static final Logger log = LoggerFactory.getLogger(Zhaobiao.class);
	String charset="utf-8";
	public static void main(String[] args) throws Exception {
		
		
		//文档编码格式为gbk
		Zhaobiao app = new Zhaobiao();//又是这种写法，构建实现类    
		
//		app.tokenizerFactory.
		
		app.makeParagraphVectors();     //调用构建模型方法      
		app.checkUnlabeledData();       //检查标签数据 
		/*
		 * Your output should be like this:
		 * 
		 * Document 'health' falls into the following categories: health:
		 * 0.29721372296220205 science: 0.011684473733853906 finance:
		 * -0.14755302887323793
		 * 
		 * Document 'finance' falls into the following categories: health:
		 * -0.17290237675941766 science: -0.09579267574606627 finance:
		 * 0.4460859189453788
		 * 
		 * so,now we know categories for yet unseen documents
		 */
	}

	
	void makeParagraphVectors() throws Exception {
		ClassPathResource resource = new ClassPathResource("0418招投标训练集/labeled");//弄一个带标签的文档路径    

		// build a iterator for our dataset
		//实现LabelAwareIterator接口，添加数据源，构成迭代器           
		iterator = new GBKFileLabelAwareIterator.Builder(charset).addSourceFolder(resource.getFile()).build();

		tokenizerFactory = new HanlpTokenizerFactory();//构建中文分割器    
		tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

		// ParagraphVectors training configuration
		
		//ParagraphVectors继承Word2Vec，Word2Vec继承SequenceVectors，
		//配置ParagraphVectors的学习率，最小学习率，批大小，步数，迭代器，同时构建词和文档，词分割器
		paragraphVectors = new ParagraphVectors.Builder()
				.learningRate(0.025)
				.minLearningRate(0.001)
				.batchSize(8000)
				.epochs(20)
				.iterate(iterator)
				.trainWordVectors(true)
				.tokenizerFactory(tokenizerFactory).build();
		
		
		//模型定型
		// Start model training
		paragraphVectors.fit();
	}

	void checkUnlabeledData() throws FileNotFoundException {
		
		//这里假定模型已经构建好，现在预测无标签的文档属于哪个类，我们装载无标签文档并对其进行检测      
		/*
		 * At this point we assume that we have model built and we can check
		 * which categories our unlabeled document falls into. So we'll start
		 * loading our unlabeled documents and checking them
		 */
		
		//构建无标签文档读取器     
		ClassPathResource unClassifiedResource = new ClassPathResource("0418招投标训练集/unlabeled");
		GBKFileLabelAwareIterator unClassifiedIterator = new GBKFileLabelAwareIterator.Builder(charset)
				.addSourceFolder(unClassifiedResource.getFile()).build();

		//预测未标记文档，很多情况一个文档可能对应多个类别，只不过每个类别值有高有低 
		/*
		 * Now we'll iterate over unlabeled data, and check which label it could
		 * be assigned to Please note: for many domains it's normal to have 1
		 * document fall into few labels at once, with different "weight" for
		 * each.
		 */
		//构建了求质心的类 
		//通过获取WordVectors实现类WordVectorsImpl中的getLookupTable方法获取查询table及tokenizerFactory构造MeansBuilder类
		MeansBuilder meansBuilder = new MeansBuilder((InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(),
				tokenizerFactory);
		
		//同理通过获取WordVectors实现类WordVectorsImpl中的getLookupTable方法获取查询table及标签列表构造LabelSeeker类    
		LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
				(InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());
		
//		paragraphVectors.getLookupTable();
		int all=0;
		int correctCount=0;
		//遍历未分类文档  
		while (unClassifiedIterator.hasNextDocument()) {
			all++;
			LabelledDocument document = unClassifiedIterator.nextDocument();
			//把文档转成向量 
			INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
			//获取文档的类别得分  
			List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

			/*
			 * please note, document.getLabel() is used just to show which
			 * document we're looking at now, as a substitute for printing out
			 * the whole document name. So, labels on these two documents are
			 * used like titles, just to visualize our classification done
			 * properly
			 */
			log.info("Document '" + document.getLabels().get(0) + "' falls into the following categories: ");
			for (Pair<String, Double> score : scores) {
				log.info("        " + score.getFirst() + ": " + score.getSecond());
			}
			String preditcedLabel = "";
			preditcedLabel =  scores.get(0).getSecond() > scores.get(1).getSecond()?scores.get(0).getFirst():scores.get(1).getFirst();
			
			if(preditcedLabel.equals(document.getLabels().get(0))){
				correctCount++;
			}
		}

		System.out.println(correctCount);
		System.out.println(all);
		System.out.println(correctCount/(double)all);
	}
}
